import csv
import re
from pathlib import Path
from statistics import median

def parse_drilling_data(pdf_path: Path, *, dpi=300, mining_area=None):
    """Render candidate PDF pages, recognize table crops, and overwrite the CSVs.

    Crops are retained in ``<mining_area>/table_images`` for inspection.
    The PDF filename stem supplies the mining area when it is not specified.
    """
    try:
        import pymupdf
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Install the parser dependencies with: pip install -r requirements-parser.txt"
        ) from exc

    # dots per inch
    if dpi < 150:
        raise ValueError("Use at least 150 DPI so table text remains readable.")
    pdf_path = Path(pdf_path)
    output_dir = pdf_path.with_name(f"{mining_area or pdf_path.stem}")
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "table_images"
    image_dir.mkdir(exist_ok=True)
    cv2, np, _ = _image_dependencies()

    assays = []
    collars = []
    table_pages = []

    with pymupdf.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text()
            # Text is only a cheap page filter; it never supplies table rows.
            has_large_scan = any(
                pymupdf.Rect(info["bbox"]).get_area() >= page.rect.get_area() * 0.25
                for info in page.get_image_info()
            )
            if text.strip() and not _is_drilling_header(text) and not has_large_scan:
                continue
            pixmap = page.get_pixmap(dpi=dpi, alpha=False, colorspace=pymupdf.csRGB)
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
            regions = _table_regions(gray)
            if not regions:
                regions = [(0, 0, image.width, image.height)]
            found = False
            for table_number, box in enumerate(regions, start=1):
                crop_path = image_dir / f"page_{page_number:03d}_table_{table_number:02d}.png"
                image.crop(box).save(crop_path)
                table = recognize_table(crop_path)
                if not table["header"]:
                    crop_path.unlink()
                    continue
                _restore_hole_ids(table, page.get_text("words"))
                parsed = _parse_with_header(table["header"], table["rows"])
                if parsed is None:
                    continue
                found = True
                table_type, rows = parsed
                if table_type == "assays":
                    assays.extend(rows)
                elif table_type == "collars":
                    collars.extend(rows)
            if found:
                table_pages.append(page_number)

    _write_csv(
        output_dir / "raw_assays.csv",
        ["hole_id", "from_m", "to_m", "width_m", "au_gpt"],
        assays,
    )
    _write_csv(
        output_dir / "raw_collars.csv",
        ["hole_id", "easting", "northing", "elevation", "dip", "azimuth", "total_depth"],
        collars,
    )

    return {
        "assays": output_dir / "raw_assays.csv",
        "collars": output_dir / "raw_collars.csv",
        "assay_rows": len(assays),
        "collar_rows": len(collars),
        "table_pages": table_pages,
    }


def recognize_table(image_path: Path, *, language="eng"):
    """Return ``{'header': list[str], 'rows': list[list[str]]}`` from a crop.

    OpenCV locates grid boundaries and Tesseract reads individual cells.
    Borderless tables use OCR word alignment instead. An image without a
    recognized drilling header returns empty lists. No values are inferred.
    """
    from PIL import Image, ImageOps

    cv2, np, _ = _image_dependencies()
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    horizontal, vertical = _line_masks(gray)
    xs = _line_positions(vertical, axis=0)
    ys = _line_positions(horizontal, axis=1)
    if len(xs) >= 3:
        xs = _with_edges(xs, image.width, gray, axis=0)
        ys = _with_edges(ys, image.height, gray, axis=1)
        if len(ys) >= 3:
            original_image = image
            pixels = np.asarray(image).copy()
            rules = cv2.dilate(cv2.bitwise_or(horizontal, vertical), None, iterations=1)
            pixels[rules > 0] = 255
            clean_gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
            # Shaded empty cells must not become OCR glyphs after rule removal.
            _, clean_gray = cv2.threshold(clean_gray, 180, 255, cv2.THRESH_BINARY)
            image = Image.fromarray(clean_gray)
            rows = []
            header_found = False
            for top, bottom in zip(ys, ys[1:]):
                if bottom - top < 6:
                    continue
                row = []
                for left, right in zip(xs, xs[1:]):
                    # A small inset removes rules without erasing cell text.
                    source = image if header_found else original_image
                    cell = source.crop((left + 2, top + 2, right - 2, bottom - 2))
                    row.append(_ocr_text(cell, language=language))
                rows.append(row)
                if not header_found:
                    header_found = bool(_split_header(rows)["header"])
                    if len(rows) >= 8 and not header_found:
                        return {"header": [], "rows": []}
            return _split_header(rows)
    return _borderless_table(image, language=language)


def _image_dependencies():
    try:
        import cv2
        import numpy as np
        import pytesseract
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Install the parser dependencies with: pip install -r requirements-parser.txt"
        ) from exc
    return cv2, np, pytesseract


def _ocr_text(image, *, language="eng"):
    _, _, pytesseract = _image_dependencies()
    try:
        return _clean_cell(pytesseract.image_to_string(image, lang=language, config="--psm 6"))
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError("Tesseract OCR is required. On macOS: brew install tesseract") from exc


def _line_masks(gray):
    cv2, _, _ = _image_dependencies()
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )
    height, width = gray.shape
    horizontal = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, width // 20), 1)),
    )
    vertical = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(50, height // 30))),
    )
    return horizontal, vertical


def _table_regions(gray):
    cv2, _, _ = _image_dependencies()
    horizontal, vertical = _line_masks(gray)
    grid = cv2.dilate(cv2.bitwise_or(horizontal, vertical), None, iterations=2)
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = gray.shape
    regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < width * 0.25 or h < 60:
            continue
        if len(_line_positions(vertical[y:y+h, x:x+w], axis=0)) < 3:
            continue
        regions.append((max(0, x-4), max(0, y-4), min(width, x+w+4), min(height, y+h+4)))
    return sorted(regions, key=lambda box: (box[1], box[0]))


def _line_positions(mask, *, axis):
    _, np, _ = _image_dependencies()
    counts = np.count_nonzero(mask, axis=axis)
    threshold = max(15, mask.shape[axis] * 0.2)
    positions = np.flatnonzero(counts >= threshold)
    if not len(positions):
        return []
    groups = np.split(positions, np.flatnonzero(np.diff(positions) > 2) + 1)
    return [int(np.median(group)) for group in groups]


def _with_edges(positions, size, gray, *, axis):
    _, np, _ = _image_dependencies()
    positions = list(positions)
    ink = np.count_nonzero(gray < 200, axis=axis)
    if positions[0] > 10 and ink[:positions[0]-3].sum() > 20:
        positions.insert(0, 0)
    if positions[-1] < size - 10 and ink[positions[-1]+4:].sum() > 20:
        positions.append(size)
    return positions


def _split_header(rows):
    rows = [_clean_row(row) for row in rows if any(row)]
    for index in range(min(8, len(rows))):
        for count in (1, 2):
            if index + count > len(rows):
                continue
            header = rows[index]
            if count == 2:
                if any(_number(cell) is not None for cell in rows[index + 1]):
                    continue
                header = _merge_header_rows(header, rows[index + 1])
            if _valid_table_header(header):
                return {"header": header, "rows": rows[index + count:]}
    return {"header": [], "rows": []}


def _is_drilling_header(text):
    return _is_assay_header(text) or _is_collar_header(text)


def _valid_table_header(header):
    text = " ".join(header)
    if not _is_drilling_header(text):
        return False
    kind = "assays" if _is_assay_header(text) else "collars"
    columns = _map_columns(header, kind)
    required = ("hole_id", "from_m", "to_m", "width_m") if kind == "assays" else tuple(columns)
    indices = [columns[key] for key in required]
    return min(indices) >= 0 and len(set(indices)) == len(indices)


def _restore_hole_ids(table, pdf_words):
    """Resolve only unique OCR lookalike matches against native PDF spellings."""
    def key(w):
        return re.sub(r"[^A-Z0-9]", "", w.upper()).replace("O", "0")
    spellings = {}
    for word in pdf_words:
        value = word[4]
        if re.fullmatch(r"[A-Za-z0-9_-]+", value) and re.search(r"[A-Za-z]", value) and re.search(r"\d", value):
            spellings.setdefault(key(value), set()).add(value)
    kind = "assays" if _is_assay_header(" ".join(table["header"])) else "collars"
    index = _map_columns(table["header"], kind)["hole_id"]
    for row in table["rows"]:
        recognized = key(_value(row, index))
        matches = spellings.get(recognized, set())
        if not matches and len(recognized) >= 4:
            # OCR occasionally duplicates a rounded glyph in a numeric suffix.
            for native, values in spellings.items():
                if re.sub(r"0+", "0", recognized) == re.sub(r"0+", "0", native):
                    matches = matches | values
        if len(matches) == 1:
            row[index] = next(iter(matches))


def _borderless_table(image, *, language):
    _, _, pytesseract = _image_dependencies()
    try:
        data = pytesseract.image_to_data(
            image, lang=language, config="--psm 6", output_type=pytesseract.Output.DICT
        )
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError("Tesseract OCR is required. On macOS: brew install tesseract") from exc
    words = []
    for i, text in enumerate(data["text"]):
        if text.strip() and float(data["conf"][i]) >= 0:
            words.append((data["left"][i], data["top"][i], data["width"][i], data["height"][i], text))
    if not words:
        return {"header": [], "rows": []}
    lines = []
    tolerance = median(word[3] for word in words) * 0.6
    for word in sorted(words, key=lambda w: (w[1] + w[3]/2, w[0])):
        center = word[1] + word[3]/2
        if not lines or abs(center - median(w[1]+w[3]/2 for w in lines[-1])) > tolerance:
            lines.append([])
        lines[-1].append(word)
    lines = [sorted(line, key=lambda word: word[0]) for line in lines]
    for index in range(len(lines)):
        for count in (1, 2, 3):
            header_words = [word for line in lines[index:index+count] for word in line]
            text = " ".join(w[4] for w in header_words)
            if not _is_drilling_header(text):
                continue
            # Persistent gaps across header lines define the column boundaries.
            spans = sorted((w[0], w[0]+w[2]) for w in header_words)
            clusters = []
            gap = median(w[3] for w in header_words) * 0.7
            for left, right in spans:
                if clusters and left - clusters[-1][1] < gap:
                    clusters[-1][1] = max(clusters[-1][1], right)
                else:
                    clusters.append([left, right])
            boundaries = [0] + [(a[1]+b[0])/2 for a, b in zip(clusters, clusters[1:])] + [image.width]
            def cells(line):
                row = []
                for lt, rt in zip(boundaries, boundaries[1:]):
                    selected = [w for w in line if lt <= w[0]+w[2]/2 < rt]
                    row.append(" ".join(w[4] for w in sorted(selected, key=lambda w: (w[1], w[0]))))
                return row
            header = cells(header_words)
            if not _valid_table_header(header):
                continue
            return {"header": header, "rows": [cells(line) for line in lines[index+count:]]}
    return {"header": [], "rows": []}


def _parse_table(table):
    rows = [_clean_row(row) for row in table or []]
    rows = [row for row in rows if any(row)]

    for header_idx in range(min(len(rows), 8)):
        header = rows[header_idx]
        parsed = _parse_with_header(header, rows[header_idx + 1 :])
        if parsed is not None:
            return parsed

        if header_idx + 1 >= len(rows):
            continue

        header = _merge_header_rows(rows[header_idx], rows[header_idx + 1])
        parsed = _parse_with_header(header, rows[header_idx + 2 :])
        if parsed is not None:
            return parsed

    return None


def _parse_with_header(header, data_rows):
    header_text = " ".join(header)
    if _is_assay_header(header_text):
        column_map = _map_columns(header, "assays")
        return "assays", _parse_assay_rows(data_rows, column_map)
    if _is_collar_header(header_text):
        column_map = _map_columns(header, "collars")
        return "collars", _parse_collar_rows(data_rows, column_map)

    return None


def _merge_header_rows(top_row, bottom_row):
    width = max(len(top_row), len(bottom_row))
    merged = []
    for idx in range(width):
        top = _value(top_row, idx)
        bottom = _value(bottom_row, idx)
        merged.append(" ".join(part for part in (top, bottom) if part))
    return merged


def _clean_row(row):
    return [_clean_cell(cell) for cell in row or []]


def _clean_cell(value) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\n", " ")).strip()


def _normalize(value):
    return re.sub(r"[^a-z0-9]+", " ", _clean_cell(value).lower()).strip()


def _is_assay_header(header_text):
    normalized = f" {_normalize(header_text)} "
    return (
        _contains_hole_id(normalized)
        and " from " in normalized
        and " to " in normalized
        and any(term in normalized for term in (" interval ", " width "))
    )


def _is_collar_header(header_text):
    normalized = f" {_normalize(header_text)} "
    return (
        _contains_hole_id(normalized)
        and " easting " in normalized
        and " northing " in normalized
        and any(term in normalized for term in (" elevation ", " elev ", " rl "))
        and any(term in normalized for term in (" azimuth ", " azi "))
        and " dip " in normalized
        and any(term in normalized for term in (" total depth ", " max depth ", " final depth ", " depth "))
    )


def _contains_hole_id(normalized_text):
    return any(
        term in normalized_text
        for term in (" hole id ", " drill hole ", " drillhole ")
    )


def _map_columns(header, table_type)->dict[str, int]:
    normalized = [_normalize(cell) for cell in header]

    if table_type == "assays":
        return {
            "hole_id": _find_column(
                normalized, ("hole id", "drill hole", "drillhole")
            ),
            "from_m": _find_column(normalized, ("from",)),
            "to_m": _find_column(normalized, ("to",)),
            "width_m": _find_column(normalized, ("interval", "width")),
            "au_gpt": _find_column(normalized, ("au g t", "au gpt", "au ppm", "au", "gold")),
        }

    return {
        "hole_id": _find_column(
            normalized, ("hole id", "drill hole", "drillhole")
        ),
        "easting": _find_column(normalized, ("easting", "east", "m e")),
        "northing": _find_column(normalized, ("northing", "north", "m n")),
        "elevation": _find_column(normalized, ("elevation", "elev", "rl")),
        "dip": _find_column(normalized, ("dip",)),
        "azimuth": _find_column(normalized, ("azimuth", "azi", "bearing")),
        "total_depth": _find_column(normalized, ("total depth", "max depth", "final depth", "depth"), exclude=("start",)),
    }


def _find_column(normalized_header, candidates, *, exclude=()) -> int:
    for candidate in candidates:
        candidate = _normalize(candidate)
        for idx, column in enumerate(normalized_header):
            if any(term in column.split() for term in exclude):
                continue
            if f" {candidate} " in f" {column} ":
                return idx
    return -1


def _parse_assay_rows(rows, column_map):
    parsed_rows = []
    required = ("hole_id", "from_m", "to_m", "width_m")
    if any(column_map.get(field, -1) < 0 for field in required):
        return parsed_rows

    cur_hole_id = ""
    cur_from = ""
    cur_to = ""
    for row in rows:
        if _looks_like_repeated_header(row):
            continue

        hole_id = _value(row, column_map["hole_id"])
        from_m = _number(_value(row, column_map["from_m"]))
        to_m = _number(_value(row, column_map["to_m"]))
        width_m = _number(_value(row, column_map["width_m"]))
        au_gpt = _number(_value(row, column_map["au_gpt"]))

        if not hole_id:
            hole_id = cur_hole_id

        cur_hole_id = hole_id

        if not width_m:
            continue

        if not from_m:
            from_m = cur_from
            to_m = cur_to

        cur_from = from_m
        cur_to = to_m
        parsed_rows.append(
            {
                "hole_id": hole_id,
                "from_m": from_m,
                "to_m": to_m,
                "width_m": width_m,
                "au_gpt": au_gpt,
            }
        )

    return parsed_rows


def _parse_collar_rows(rows, column_map):
    parsed_rows = []
    required = ("hole_id", "easting", "northing", "elevation", "dip", "azimuth", "total_depth")
    if any(column_map.get(field, -1) < 0 for field in required):
        return parsed_rows

    for row in rows:
        if _looks_like_repeated_header(row):
            continue

        hole_id = _value(row, column_map["hole_id"])
        if not hole_id:
            continue

        parsed_row = {"hole_id": hole_id}
        for field in required[1:]:
            fv = _value(row, column_map[field])
            parsed_row[field] = _number(fv)

        if parsed_row["easting"] is None or parsed_row["northing"] is None:
            continue

        parsed_rows.append(parsed_row)

    return parsed_rows


def _value(row: list[str], idx: int) -> str:
    if idx < 0 or idx >= len(row):
        return ""
    return _clean_cell(row[idx])


def _number(value: str) -> str | None:
    value = _clean_cell(value)
    if not value:
        return None

    match = re.fullmatch(r"[<>\u2264\u2265]?\s*([+-]?(?:\d[\d,]*(?:\.\d*)?|\.\d+))", value)
    if not match:
        return None

    number = match.group(1).replace(",", "")
    return number


def _looks_like_repeated_header(row):
    text = " ".join(row)
    return _is_assay_header(text) or _is_collar_header(text)


def _write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

if __name__ == '__main__':
    result = parse_drilling_data(Path("raw_data/16062026_south_emu-triton_reedy_val.pdf"))
    print(result)