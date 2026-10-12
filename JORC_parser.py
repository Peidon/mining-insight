import csv
import re
from pathlib import Path

"""
Parse the data of mineral assays and mining hole collars from PDF files.
Save raw_assays.csv and raw_collars.csv at `{mining_area}_data` directory, {mining_area} is name of mining area.
If there already exists the csv files, just overwrite.
Step 1, locate the data:
For mining hole collars, find tables which has header that includes "hold id, easting, northing, elevation, start depth, max depth/total depth, azimuth, dip".
For mineral assays, find tables which has header that includes "hold id, from, to, interval".
step2, Parse these tables, then save data into the csv files.
Header of raw_assays.csv is "hole_id,from_m,to_m,width_m,au_gpt",
header of raw_collars.csv is "hole_id,easting,northing,elevation,dip,azimuth,total_depth".
"""


def parse_drilling_data(pdf_path: Path):
    try:
        import pdfplumber
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "parse_drilling_data requires pdfplumber to extract tables from PDFs."
        ) from exc

    pdf_path = Path(pdf_path)
    output_dir = pdf_path.with_name(f"{pdf_path.stem}")
    output_dir.mkdir(parents=True, exist_ok=True)

    assays = []
    collars = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables() or []:
                parsed = _parse_table(table)
                if parsed is None:
                    continue

                table_type, rows = parsed
                if table_type == "assays":
                    assays.extend(rows)
                elif table_type == "collars":
                    collars.extend(rows)

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
    }


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
        for term in (" hole id ", " holeid ", " drill hole ", " drillhole ")
    )


def _map_columns(header, table_type)->dict[str, int]:
    normalized = [_normalize(cell) for cell in header]

    if table_type == "assays":
        return {
            "hole_id": _find_column(
                normalized, ("hole id", "holeid", "drill hole", "drillhole")
            ),
            "from_m": _find_column(normalized, ("from",)),
            "to_m": _find_column(normalized, ("to",)),
            "width_m": _find_column(normalized, ("interval", "width")),
            "au_gpt": _find_column(normalized, ("au g t", "au gpt", "au ppm", "au", "gold")),
        }

    return {
        "hole_id": _find_column(
            normalized, ("hole id", "holeid", "drill hole", "drillhole")
        ),
        "easting": _find_column(normalized, ("easting", "east", "m e")),
        "northing": _find_column(normalized, ("northing", "north", "m n")),
        "elevation": _find_column(normalized, ("elevation", "elev", "rl")),
        "dip": _find_column(normalized, ("dip",)),
        "azimuth": _find_column(normalized, ("azimuth", "azi", "bearing")),
        "total_depth": _find_column(normalized, ("total depth", "max depth", "final depth", "depth")),
    }


def _find_column(normalized_header, candidates) -> int:
    for candidate in candidates:
        candidate = _normalize(candidate)
        for idx, column in enumerate(normalized_header):
            if column == candidate or candidate in column:
                return idx
    return -1


def _parse_assay_rows(rows, column_map):
    parsed_rows = []
    required = ("hole_id", "from_m", "to_m", "width_m")
    if any(column_map.get(field) is None for field in required):
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

        if not from_m:
            from_m = cur_from
            to_m = cur_to

        if not width_m:
            continue

        cur_hole_id = hole_id
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
    if any(column_map.get(field) is None for field in required):
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

    match = re.search(r"-?\d[\d,]*(?:\.\d+)?", value)
    if not match:
        return None

    number = match.group(0).replace(",", "")
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
    result = parse_drilling_data(Path("raw_data/31082026_south_emu-triton_reedy_val.pdf"))
    print(result)