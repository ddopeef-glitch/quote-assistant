"""Build fictional fixtures from code only. Never read business workbooks."""
from pathlib import Path
import sys

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core import normalize_specification

HEADERS = ["分类", "源代码", "商标", "品名", "规格", "别名", "价格标记", "轻重", "单价", "包装数",
           "毛坯单重", "成品单重", "纯毛单重", "启用状态", "记录键", "备注"]
# All prices, quantities, weights, codes and alias relationships are invented.
PRODUCTS = [
    ("沟槽", "90°弯头", "76", "b", "90弯头；弯头；沟槽弯头"),
    ("沟槽", "90°弯头", "108", "b", "90弯头；弯头；沟槽弯头"),
    ("沟槽", "90°弯头", "114", "b", "90弯头；弯头；沟槽弯头"),
    ("沟槽", "90°弯头", "11422.5", "b", "90弯头；弯头"),
    ("沟槽", "90°弯头", "16522.5", "b", "90弯头；弯头"),
    ("沟槽", "卡箍", "76", "b", ""),
    ("玛钢", "弯头", "25", "a", "90°弯头；90弯头"),
    ("玛钢", "弯头", "25", "b", "90°弯头；90弯头"),
    ("玛钢", "弯头", "65", "a", "90°弯头；90弯头"),
    ("玛钢", "弯头", "65", "A", "90°弯头；90弯头"),
    ("玛钢", "弯头", "65", "b", "90°弯头；90弯头"),
    ("玛钢", "三通", "25", "a", "T"),
    ("玛钢", "三通", "25", "b", "T"),
    ("玛钢", "三通", "32x25x25", "a", "T"),
    ("玛钢", "三通", "50", "b", "T"),
    ("玛钢", "三通", "40x25", "b", "T"),
    ("玛钢", "三通", "40x25x32", "b", "T"),
    ("玛钢", "三通", "50x20", "a", "T"),
    ("玛钢", "异径三通", "50x20", "a", "T"),
    ("玛钢", "管古", "25", "b", "管箍"),
    ("玛钢", "球铁弯头", "25", "b", ""),
    ("玛钢", "球铁三通", "25", "b", ""),
]


def make_book(path, *, empty=False):
    workbook = Workbook()
    workbook.remove(workbook.active)
    workbook.properties.creator = "Synthetic Demo"
    workbook.properties.lastModifiedBy = "Synthetic Demo"
    workbook.properties.title = "虚构演示数据，不可用于商业报价"
    sheets = {}
    for name in ("沟槽", "玛钢"):
        sheet = workbook.create_sheet(name + "数据")
        sheets[name] = sheet
        sheet.append(["虚构演示主数据 — 所有数值仅用于测试"])
        sheet.merge_cells("A1:P1")
        sheet.cell(3, 1, "合成数据；非真实商品价格、包装或企业库存。")
        sheet.append([])
        for col, header in enumerate(HEADERS, 1):
            sheet.cell(5, col, header)
        sheet.freeze_panes = "D6"
        sheet.auto_filter.ref = "A5:P" + str(5 if empty else 40)
        validation = DataValidation(type="list", formula1='"a,A,b"')
        sheet.add_data_validation(validation)
        validation.add("G6:G1000")
        for column in "ABCDEFGHIJKLMNOP":
            sheet.column_dimensions[column].width = 15
        sheet.column_dimensions["F"].width = 30
        sheet.column_dimensions["O"].width = 36
        sheet.column_dimensions["P"].width = 48
    if not empty:
        for index, (category, name, spec, marker, aliases) in enumerate(PRODUCTS, 1):
            pending = category == "沟槽" and spec.endswith("22.5")
            missing_pack = name == "球铁三通"
            notes = "全新虚构测试数据"
            if pending:
                notes += "；缺面价，禁止报价"
            if missing_pack:
                notes += "；缺包装数"
            sheets[category].append([
                category, f"DEMO-{index:03}", "DEMO", name, spec, aliases, marker,
                {"a": "轻货", "A": "", "b": "重货"}[marker],
                None if pending else 10 + index, None if missing_pack else 10,
                100 + index, 90 + index, 80 + index,
                "待补充" if pending else "启用",
                f"{category}｜{name}｜{normalize_specification(spec)}｜{marker}", notes,
            ])
    grades = workbook.create_sheet("等级映射")
    grades.append(["虚构演示规则：等级与别名"])
    for col, title in enumerate(("客户写法", "价格标记", "启用状态", "备注"), 1):
        grades.cell(4, col, title)
    for row, (raw, marker) in enumerate((("蓝袋", "b"), ("好", "b"), ("黑袋", "a"), ("中", "a")), 5):
        for col, value in enumerate((raw, marker, "启用", "演示约定"), 1):
            grades.cell(row, col, value)
    for col, title in enumerate(("分类", "别名", "正式品名", "启用状态", "备注"), 6):
        grades.cell(13, col, title)
    for row, (alias, target) in enumerate((("三通", "三通"), ("三通", "异径三通"),
                                          ("管古", "管箍"), ("管古", "异径管箍")), 14):
        for col, value in enumerate(("玛钢", alias, target, "启用", "合成别名测试关系"), 6):
            grades.cell(row, col, value)
    for sheet in workbook:
        for row in sheet:
            for cell in row:
                cell.font = Font(name="Calibri", size=11)
                cell.alignment = Alignment(vertical="center")
        for cell in sheet[5 if sheet != grades else 4]:
            cell.font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="203A52")
    path.parent.mkdir(parents=True, exist_ok=True)
    # A setup rerun must never replace a user's edited workbook.
    with path.open("xb") as handle:
        workbook.save(handle)
    workbook.close()


def main():
    for path, empty in ((ROOT / "主数据/标准产品价格表.xlsx", False),
                        (ROOT / "模板/标准产品价格表_空白模板.xlsx", True)):
        if path.exists():
            print("Preserved existing:", path.relative_to(ROOT))
        else:
            make_book(path, empty=empty)
            print("Created synthetic:", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
