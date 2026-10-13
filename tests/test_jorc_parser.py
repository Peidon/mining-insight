import csv
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import pymupdf

from JORC_parser import (
    _map_columns, _parse_table, _restore_hole_ids, _valid_table_header,
    parse_drilling_data, recognize_table,
)


@unittest.skipUnless(shutil.which("tesseract"), "Tesseract OCR is required")
class TableRecognitionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def table_image(self, *, ruled=True):
        image = Image.new("RGB", (1100, 320), (235, 235, 235))
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=26)
        xs = [20, 300, 490, 680, 870, 1080]
        ys = [20, 100, 160, 220, 300]
        data = [
            ["Hole ID", "From (m)", "To (m)", "Interval (m)", "Au (g/t)"],
            ["DD123", "0.0", "2.0", "2.0", "1.25"],
            ["", "2.0", "3.5", "1.5", "2.75"],
            ["DD234", "10.0", "11.0", "1.0", "0.50"],
        ]
        if ruled:
            for x in xs:
                draw.line((x, ys[0], x, ys[-1]), fill="black", width=2)
            for y in ys:
                draw.line((xs[0], y, xs[-1], y), fill="black", width=2)
        for row_index, row in enumerate(data):
            for col_index, value in enumerate(row):
                draw.text((xs[col_index]+12, ys[row_index]+20), value, fill="black", font=font)
        path = self.root / "table.png"
        image.save(path)
        return path

    def test_ruled_cells_and_blank_id(self):
        table = recognize_table(self.table_image())
        self.assertEqual(len(table["header"]), 5)
        self.assertEqual(len(table["rows"]), 3)
        self.assertEqual(table["rows"][1][0], "")
        kind, rows = _parse_table([table["header"], *table["rows"]])
        self.assertEqual(kind, "assays")
        self.assertEqual(rows[0]["from_m"], "0.0")
        self.assertEqual(rows[1]["hole_id"], "DD123")
        self.assertEqual(rows[2]["au_gpt"], "0.50")

    def test_borderless_alignment(self):
        table = recognize_table(self.table_image(ruled=False))
        self.assertEqual(len(table["header"]), 5)
        self.assertEqual(len(table["rows"]), 3)
        self.assertEqual(table["rows"][1][0], "")

    def test_blank_image(self):
        path = self.root / "blank.png"
        Image.new("RGB", (500, 500), "white").save(path)
        self.assertEqual(recognize_table(path), {"header": [], "rows": []})

    def test_scanned_pdf_and_overwrite(self):
        image_path = self.table_image()
        pdf_path = self.root / "mine.pdf"
        with pymupdf.open() as doc:
            page = doc.new_page(width=550, height=160)
            page.insert_image(page.rect, filename=str(image_path))
            # A native footer must not cause the scanned table to be skipped.
            page.insert_text((10, 155), "Report footer", fontsize=6)
            doc.save(pdf_path)
        output = self.root / "mine_data"
        output.mkdir()
        (output / "raw_assays.csv").write_text("stale data")
        result = parse_drilling_data(pdf_path)
        self.assertEqual(result["table_pages"], [1])
        self.assertEqual(result["assay_rows"], 3)
        self.assertEqual(result["collar_rows"], 0)
        with result["assays"].open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[0]["hole_id"], "DD123")
        self.assertEqual(rows[0]["from_m"], "0.0")
        # self.assertTrue(list((output / "table_images").glob("*.png")))


class DrillingParsingTests(unittest.TestCase):
    def test_jorc_prose_is_not_a_table_header(self):
        self.assertFalse(_valid_table_header([
            "Drill hole information",
            "Easting northing elevation dip azimuth total depth",
            "Commentary",
        ]))

    def test_hole_id_corrections_require_unique_native_spelling(self):
        table = {"header": ["Hole ID", "From", "To", "Interval", "Au"],
                 "rows": [["26REDDOO0O1 W1", "1", "2", "1", "3"],
                          ["DDO1", "1", "2", "1", "3"]]}
        words = [(0, 0, 1, 1, value) for value in ("26REDD001_W1", "DD01", "DDO1")]
        _restore_hole_ids(table, words)
        self.assertEqual(table["rows"][0][0], "26REDD001_W1")
        self.assertEqual(table["rows"][1][0], "DDO1")

    def test_total_depth_does_not_use_start_depth(self):
        header = ["Hold ID", "Easting", "Northing", "Elevation", "Start depth", "Max depth", "Azimuth", "Dip"]
        _, rows = _parse_table([header, ["DD001", "500,100", "6,200,200", "450", "10", "123.4", "90", "-60"]])
        self.assertEqual(rows[0]["total_depth"], "123.4")
        self.assertEqual(rows[0]["dip"], "-60")
        self.assertEqual(rows[0]["easting"], "500100")
        self.assertEqual(_map_columns(["Start depth"], "collars")["total_depth"], -1)

    def test_missing_columns_cannot_use_last_cell(self):
        table = [["Hole ID From To Interval", "Au"], ["DD001", "2.3"]]
        self.assertEqual(_parse_table(table)[1], [])

    def test_included_interval_does_not_borrow_parent_depths(self):
        table = [
            ["Hole ID", "From", "To", "Interval", "Au"],
            ["DD001", "0", "10", "10", "1.2"],
            ["", "", "incl.", "2", "5.0"],
            ["DD002", "NSI", "", "", ""],
            ["", "20", "21", "1", "2.0"],
        ]
        _, rows = _parse_table(table)
        self.assertEqual(rows[0]["from_m"], "0")
        self.assertEqual(rows[1]["hole_id"], "DD001")
        self.assertEqual(rows[1]["from_m"], "0")
        self.assertEqual(rows[1]["to_m"], "10")
        self.assertEqual(rows[2]["hole_id"], "DD002")


if __name__ == "__main__":
    unittest.main()
