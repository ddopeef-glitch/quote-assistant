from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

from openpyxl import Workbook

from src.input_router import extract_evidence_units


def _xlsx_with_dimension(dimension, cells):
    workbook = Workbook()
    sheet = workbook.active
    for coordinate, value in cells.items():
        sheet[coordinate] = value
    source = BytesIO()
    workbook.save(source)

    output = BytesIO()
    with ZipFile(BytesIO(source.getvalue())) as source_zip, ZipFile(output, "w", ZIP_DEFLATED) as output_zip:
        for info in source_zip.infolist():
            content = source_zip.read(info.filename)
            if info.filename == "xl/worksheets/sheet1.xml":
                text = content.decode("utf-8")
                start = text.index('<dimension ref="')
                end = text.index('"', start + len('<dimension ref="'))
                text = text[:start] + f'<dimension ref="{dimension}' + text[end:]
                content = text.encode("utf-8")
            output_zip.writestr(info, content)
    return output.getvalue()


def test_too_small_declared_dimension_keeps_all_nonempty_rows_and_columns():
    data = _xlsx_with_dimension(
        "A1:A1",
        {"A1": "标题", "B2": "实际询价", "D2": 8, "C4": "备注"},
    )

    units = extract_evidence_units("synthetic.xlsx", data)

    assert [(unit["row"], unit["text"]) for unit in units] == [
        (1, "A=标题"),
        (2, "B=实际询价\tD=8"),
        (4, "C=备注"),
    ]


def test_oversized_declared_dimension_does_not_expand_blank_output():
    data = _xlsx_with_dimension(
        "A1:XFD1048576",
        {"A1": "标题", "B2": "实际询价"},
    )

    units = extract_evidence_units("synthetic.xlsx", data)

    assert [(unit["row"], unit["text"]) for unit in units] == [
        (1, "A=标题"),
        (2, "B=实际询价"),
    ]
