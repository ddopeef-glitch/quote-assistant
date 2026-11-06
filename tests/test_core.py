from io import BytesIO
from pathlib import Path
from datetime import date, datetime
from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import json
import re
import sys
from types import SimpleNamespace

import fitz
from openpyxl import Workbook, load_workbook
import pandas as pd
from PIL import Image
import pytest

from config import MASTER_DATA_FILE
from src.core import (
    _chinese_money,
    _sales_specification,
    ProductRecord,
    PriceRecord,
    apply_packing_override,
    apply_price,
    confirm_inquiry,
    discover_master_data,
    generate_quotation,
    generate_sales_id,
    load_customer_names,
    local_generation_date,
    load_master_data,
    load_master_grade_markers,
    load_master_price_records,
    load_price_records,
    master_data_fingerprint,
    match_price,
    money_sum,
    money_total,
    normalize_price_marker,
    normalize_product_name,
    normalize_specification,
    remember_customer_name,
    price_aliases,
    price_product_terms,
    render_quotation_preview,
    required_categories,
)
from src.input_router import (
    AI_REVIEW_WARNING,
    _records_from_ai,
    extract_file,
    extract_text,
    validate_image,
)
from src.ocr_pipeline import LocalOCRExtractor


@pytest.fixture(autouse=True)
def offline_paddle_mode(monkeypatch):
    # Core parser tests inject OCR results and must never load the real model.
    monkeypatch.setitem(sys.modules, "paddle", SimpleNamespace(
        base=SimpleNamespace(dygraph=SimpleNamespace(guard=lambda device: nullcontext())),
    ))


def product(**changes):
    confirmed = changes.pop("manually_confirmed", True)
    values = dict(source_type="text", source_name="test", category="玛钢", product_name="弯头", specification="25", quantity_cartons=2)
    values.update(changes)
    item = ProductRecord(**values)
    if confirmed:
        confirm_inquiry(item)
    return item


def price(marker="a", amount=1.61):
    return PriceRecord("price.xlsx", "玛钢数据", 8, "弯头", "25", marker, Decimal(str(amount)), 200, category="玛钢")


def test_recognition_confirmation_accepts_empty_price_fields():
    from app import records_from_frame

    item = product(confidence=0.75)
    frame = pd.DataFrame([item.to_dict()])[["product_name", "specification", "quantity_cartons", "confidence"]]
    converted = records_from_frame(frame, [item])[0]
    assert (converted.packing_quantity, converted.unit_price, converted.total_price) == (None, None, None)

    title_scoped = ProductRecord(
        source_type="ai", source_name="test", category="球铁",
        product_name="三通", specification="25", quantity_cartons=1,
    )
    title_frame = pd.DataFrame([title_scoped.to_dict()])[["category", "product_name", "specification", "quantity_cartons"]]
    assert records_from_frame(title_frame, [title_scoped])[0].category == "球铁"


@pytest.mark.parametrize(
    "raw,expected",
    [("蓝袋", "b"), ("好", "b"), ("黑袋", "a"), ("中", "a"), ("a", "a"), ("A", "A"), ("b", "b"), (" B ", "b")],
)
def test_grade_mapping(raw, expected):
    mapping = {"蓝袋": "b", "好": "b", "黑袋": "a", "中": "a"}
    assert normalize_price_marker(raw, mapping) == expected


def test_product_name_normalization():
    assert normalize_product_name(" DEMO卡箍 ") == "卡箍"
    assert normalize_product_name("弯头") == "90°弯头"


def test_specification_normalization():
    assert normalize_specification(" 114*76G ") == "114x76"
    assert normalize_specification("32-25-25") == "32x25x25"


def test_exact_match_and_source():
    item = product(price_marker="a")
    candidates = match_price(item, [price()], "fingerprint")
    assert len(candidates) == 1
    assert item.match_status == "matched"
    assert item.price_source_row == 8


def test_missing_marker_defaults_to_b_and_never_falls_back_to_a():
    item = product()
    assert item.price_marker == "b"
    assert match_price(item, [price()], "fingerprint") == []
    assert item.match_status == "not_found"
    assert item.unit_price is None


@pytest.mark.parametrize("marker", ["c"])
def test_default_b_does_not_turn_invalid_marker_into_b(marker):
    with pytest.raises(ValueError, match="价格标记只能是 a、A、b 或空白"):
        product(price_marker=marker)


@pytest.mark.parametrize("original_grade", ["黑袋", "A"])
def test_default_b_requires_grade_text_to_be_truly_absent(original_grade):
    item = ProductRecord(
        source_type="text", source_name="test", category="玛钢", product_name="弯头",
        specification="25", quantity_cartons=2, original_grade_text=original_grade,
    )
    with pytest.raises(ValueError, match="已识别等级/袋色原文但未确认合法 a/A/b"):
        confirm_inquiry(item)


def test_missing_marker_defaults_to_existing_b_candidate():
    item = product()
    candidates = match_price(item, [price("a"), price("b", 1.82)], "fingerprint")
    assert len(candidates) == 1
    assert item.match_status == "matched"
    assert item.unit_price == Decimal("1.82")


def test_marker_filters_candidate():
    item = product(price_marker="b")
    match_price(item, [price("a"), price("b", 1.82)], "fingerprint")
    assert item.unit_price == Decimal("1.82")


@pytest.mark.parametrize("marker,amount", [("a", 8.30), ("A", 7.98), ("b", 8.98)])
def test_markers_are_case_sensitive_and_never_fall_back(marker, amount):
    prices = [price("a", 8.30), price("A", 7.98), price("b", 8.98)]
    item = product(price_marker=marker)
    candidates = match_price(item, prices, "fingerprint")
    assert [candidate.price_marker for candidate in candidates] == [marker]
    assert item.unit_price == Decimal(str(amount))
    missing = product(price_marker=marker)
    assert match_price(missing, [candidate for candidate in prices if candidate.price_marker != marker], "fingerprint") == []
    assert missing.match_status == "not_found"
    assert missing.unit_price is None


def test_not_found():
    item = product(specification="999")
    assert match_price(item, [price()], "fingerprint") == []
    assert item.match_status == "not_found"


def test_invalid_quantity():
    item = product(quantity_cartons=0, manually_confirmed=False)
    with pytest.raises(ValueError, match="箱数必须是正整数"):
        confirm_inquiry(item)


def test_synthetic_master_bare_elbow_defaults_to_90_degree_elbow():
    item = product(category="沟槽", product_name="弯头", specification="76")
    item.price_marker = "b"
    confirm_inquiry(item)
    prices = load_master_price_records(MASTER_DATA_FILE)
    candidates = match_price(item, prices, "fingerprint")
    assert [(candidate.product_name, candidate.specification) for candidate in candidates] == [("90°弯头", "76")]
    assert "弯头" in candidates[0].aliases
    assert item.product_name == "90°弯头"
    assert item.match_status == "matched"
    assert item.unit_price == candidates[0].unit_price


def test_bare_elbow_still_requires_confirmation_when_multiple_candidates_remain():
    item = product(category="沟槽", product_name="弯头", specification="76", price_marker="b")
    prices = [
        PriceRecord("master.xlsx", "沟槽数据", 8, "90°弯头", "76", "b", Decimal("10"), 20, category="沟槽"),
        PriceRecord("master.xlsx", "沟槽数据", 9, "弯头", "76", "b", Decimal("11"), 20, category="沟槽"),
    ]
    assert len(match_price(item, prices, "fingerprint")) == 2
    assert item.product_name == "90°弯头"
    assert item.match_status == "multiple_candidates"
    assert item.unit_price is None


@pytest.mark.parametrize("specification", ["114 22.5", "165 22.5"])
def test_synthetic_pending_elbows_remain_visible_without_price(specification):
    prices = load_master_price_records(MASTER_DATA_FILE)
    item = product(category="沟槽", product_name="弯头", specification=specification, price_marker="b")
    candidates = match_price(item, prices, master_data_fingerprint(MASTER_DATA_FILE))
    assert len(candidates) == 1
    assert candidates[0].unit_price is None
    assert item.product_name == "90°弯头"
    assert item.match_status == "pending"
    assert item.unit_price is None
    assert item.total_price is None
    assert item.price_source_file
    assert "面价待补充" in item.match_message


def test_groove_product_does_not_fall_back_to_threaded_product():
    item = product(category="沟槽", product_name="沟槽机械三通", specification="76", price_marker="a")
    threaded = PriceRecord("master.xlsx", "玛钢数据", 2, "丝接机械三通", "76", "a", Decimal("8.00"), 20, category="玛钢")
    match_price(item, [threaded], "fingerprint")
    assert item.match_status == "not_found"


def test_alias_match_comes_from_master_data():
    item = product(category="沟槽", product_name="刚性卡箍", specification="76", price_marker="b")
    master = PriceRecord("master.xlsx", "沟槽数据", 6, "卡箍", "76", "b", Decimal("10.67"), 40, ("刚性卡箍",), "沟槽")
    match_price(item, [master], "fingerprint")
    assert item.match_status == "matched"
    assert item.price_source_file == "master.xlsx"


def test_total_uses_decimal():
    assert money_total("3", "7", "0.1") == Decimal("2.10")
    assert money_total(1, 1, "0.001") == Decimal("0.01")
    with pytest.raises(ValueError, match="正整数"):
        money_total("1.5", 10, 1)
    assert money_total(10**1000, 10, 1) == Decimal("1" + "0" * 1001 + ".00")
    assert money_total(1, 1, Decimal("1.23000000000000000000000000001")) == Decimal("1.24")
    assert money_sum([Decimal("10000000000000000000000000000.01")] * 100) == Decimal(
        "1000000000000000000000000000001.00"
    )


def test_text_input():
    rows = extract_text("50 弯头黑袋 20 件，25 弯头蓝袋 10 件", grade_markers={"黑袋": "a", "蓝袋": "b"})
    assert [(row.specification, row.price_marker, row.quantity_cartons) for row in rows] == [("50", "a", 20), ("25", "b", 10)]


@pytest.mark.parametrize("grade,expected", [("a", "a"), ("A", "A"), ("b", "b"), ("B", "b")])
def test_text_input_preserves_explicit_letter_grade(grade, expected):
    row = extract_text(f"25 弯头 {grade} 10件")[0]
    assert row.original_grade_text == grade
    assert row.price_marker == expected


def test_context_text_input_preserves_grade_after_quantity():
    row = extract_text("弯头\n25的10件a")[0]
    assert (row.product_name, row.specification, row.quantity_cartons) == ("90°弯头", "25", 10)
    assert (row.original_grade_text, row.price_marker) == ("a", "a")


def test_context_text_input():
    rows = extract_text("弯头\n76的10件\n89的5件")
    assert [(row.product_name, row.specification, row.quantity_cartons) for row in rows] == [("90°弯头", "76", 10), ("90°弯头", "89", 5)]


def test_text_title_context_is_not_a_product_and_does_not_misread_zhongda():
    markers = {"中": "a"}
    rows = extract_text("L中=25弯头10件\n球铁\n32三通5件\n40中大3件", grade_markers=markers)
    assert [(row.product_name, row.price_marker) for row in rows] == [
        ("90°弯头", "a"), ("三通", "a"), ("中大", "a"),
    ]
    assert [row.category for row in rows] == [None, "球铁", "球铁"]
    assert all(row.original_product_name not in {"L中", "球铁"} for row in rows)


def test_synthetic_text_sample_matching():
    prices = load_master_price_records(MASTER_DATA_FILE)
    rows = extract_file("synthetic.txt", "25弯头2箱\n25三通3箱\n999弯头1箱".encode(),
                        product_terms=price_product_terms(prices),
                        grade_markers=load_master_grade_markers(MASTER_DATA_FILE))
    assert len(rows) == 3
    for row in rows:
        assert match_price(row, prices, "fingerprint") == []
        assert row.match_status == "unconfirmed"
        row.category = "玛钢"
        confirm_inquiry(row)
        match_price(row, prices, "fingerprint")
    assert [row.match_status for row in rows] == ["matched", "matched", "not_found"]


def test_multi_target_alias_keeps_all_candidates_and_never_auto_selects():
    prices = [
        PriceRecord("m.xlsx", "玛钢数据", 6, "三通", "25", "b", Decimal("1.00"), 10, ("T",), "玛钢"),
        PriceRecord("m.xlsx", "玛钢数据", 7, "异径三通", "25", "b", Decimal("1.10"), 10, ("T",), "玛钢"),
    ]
    assert price_aliases(prices)["t"] == frozenset({"三通", "异径三通"})
    item = product(product_name="T", specification="25", price_marker="b")
    candidates = match_price(item, prices, "fingerprint")
    assert {candidate.product_name for candidate in candidates} == {"三通", "异径三通"}
    assert item.match_status == "multiple_candidates"
    assert item.unit_price is None


def test_same_source_multi_target_alias_collapses_to_confirmed_name():
    mappings = (("玛钢", "三通", ("三通", "异径三通")),)
    prices = [
        PriceRecord(
            "m.xlsx", "玛钢数据", 179, "三通", "32x25x25", "a", Decimal("3.10"), 90,
            category="玛钢", multi_target_aliases=mappings,
        )
    ]
    item = product(product_name="三通", specification="32x25x25", price_marker="a")
    candidates = match_price(item, prices, "fingerprint")
    assert [(candidate.product_name, candidate.source_row) for candidate in candidates] == [("三通", 179)]
    assert item.product_name == "三通"
    assert item.match_status == "matched"
    assert item.unit_price == Decimal("3.10")


@pytest.mark.parametrize(
    "change",
    [
        pytest.param({"source_file": "other.xlsx"}, id="source-file"),
        pytest.param({"sheet_name": "其他数据"}, id="sheet-name"),
        pytest.param({"source_row": 180}, id="source-row"),
        pytest.param({"unit_price": Decimal("3.11")}, id="unit-price"),
        pytest.param({"packing_quantity": 80}, id="packing-quantity"),
        pytest.param({"status": "待补充"}, id="pending-status"),
        pytest.param({"blank_weight": Decimal("101")}, id="blank-weight"),
        pytest.param({"finished_weight": Decimal("91")}, id="finished-weight"),
        pytest.param({"pure_weight": Decimal("81")}, id="pure-weight"),
        pytest.param({"notes": "报价事实不同"}, id="notes"),
    ],
)
def test_multi_target_alias_with_different_quote_evidence_stays_blocked(change):
    first = PriceRecord(
        "m.xlsx", "玛钢数据", 179, "三通", "32x25x25", "a", Decimal("3.10"), 90,
        ("T",), "玛钢",
    )
    second_values = {
        "source_file": "m.xlsx", "sheet_name": "玛钢数据", "source_row": 179,
        "product_name": "异径三通", "specification": "32x25x25", "price_marker": "a",
        "unit_price": Decimal("3.10"), "packing_quantity": 90, "aliases": ("T",),
        "category": "玛钢",
    }
    second_values.update(change)
    item = product(product_name="T", specification="32x25x25", price_marker="a")
    candidates = match_price(item, [first, PriceRecord(**second_values)], "fingerprint")
    assert len(candidates) == 2
    assert item.match_status == "multiple_candidates"
    assert item.unit_price is None


@pytest.mark.parametrize(
    "specification,marker,row,unit_price,packing",
    [
        ("32x25x25", "a", 13, Decimal("24"), 10),
        ("50", "b", 14, Decimal("25"), 10),
        ("40x25", "b", 15, Decimal("26"), 10),
        ("40x25x32", "b", 16, Decimal("27"), 10),
    ],
)
def test_synthetic_master_same_source_tee_aliases_are_unique(specification, marker, row, unit_price, packing):
    prices = load_master_price_records(MASTER_DATA_FILE)
    item = product(category="玛钢", product_name="三通", specification=specification, price_marker=marker)
    candidates = match_price(item, prices, "fingerprint")
    assert len(candidates) == 1
    assert candidates[0].product_name == item.product_name == "三通"
    assert item.match_status == "matched"
    assert (item.price_source_sheet, item.price_source_row, item.unit_price, item.packing_quantity) == (
        "玛钢数据", row, unit_price, packing,
    )


def test_synthetic_master_multi_target_alias_returns_all_formal_candidates():
    prices = load_master_price_records(MASTER_DATA_FILE)
    assert price_aliases(prices)["管古"] == frozenset({"管箍", "异径管箍"})
    item = product(category="玛钢", product_name="管古", specification="25", price_marker="b")
    candidates = match_price(item, prices, "fingerprint")
    assert [(candidate.product_name, candidate.sheet_name, candidate.source_row) for candidate in candidates] == [
        ("管古", "玛钢数据", 19)
    ]
    assert item.product_name == "管古"
    assert item.match_status == "matched"
    assert item.unit_price == candidates[0].unit_price


def test_category_isolation_is_part_of_matching_key():
    prices = [
        PriceRecord("m.xlsx", "沟槽数据", 6, "弯头", "25", "b", Decimal("2.00"), 10, category="沟槽"),
        PriceRecord("m.xlsx", "玛钢数据", 6, "弯头", "25", "b", Decimal("1.00"), 10, category="玛钢"),
    ]
    item = product(category="沟槽", price_marker="b")
    candidates = match_price(item, prices, "fingerprint")
    assert len(candidates) == 1
    assert item.unit_price == Decimal("2.00")
    item.category = "玛钢"
    assert match_price(item, prices, "fingerprint") == []
    assert item.match_status == "unconfirmed"


def test_ball_iron_is_a_ui_partition_of_malleable_records_and_keeps_confirmed_name():
    prices = [
        PriceRecord("m.xlsx", "沟槽数据", 6, "三通", "25", "b", Decimal("3.00"), 10, category="沟槽"),
        PriceRecord("m.xlsx", "玛钢数据", 7, "三通", "25", "b", Decimal("1.00"), 20, category="玛钢"),
        PriceRecord("m.xlsx", "玛钢数据", 8, "球热三通", "25", "b", Decimal("2.00"), 30, category="玛钢"),
        PriceRecord("m.xlsx", "玛钢数据", 9, "球铁管古", "32", "b", Decimal("4.00"), 40, category="玛钢"),
    ]

    malleable = product(category="玛钢", product_name="三通", specification="25", price_marker="b")
    assert [candidate.source_row for candidate in match_price(malleable, prices, "fingerprint")] == [7]

    ball = product(category="球铁", product_name="三通", specification="25", price_marker="b")
    assert [candidate.source_row for candidate in match_price(ball, prices, "fingerprint")] == [8]
    assert ball.product_name == "三通"

    prefixed = product(category="球铁", product_name="球铁管古", specification="32", price_marker="b")
    assert [candidate.source_row for candidate in match_price(prefixed, prices, "fingerprint")] == [9]
    assert prefixed.product_name == "球铁管古"


def test_ball_iron_uses_physical_malleable_multi_target_alias_without_crossing_ui_partition():
    mappings = (("玛钢", "三通", ("球热三通", "异径三通")),)
    prices = [
        PriceRecord(
            "m.xlsx", "玛钢数据", 8, "球热三通", "25", "b", Decimal("2.00"), 30,
            category="玛钢", multi_target_aliases=mappings,
        )
    ]
    ball = product(category="球铁", product_name="异径三通", specification="25", price_marker="b")
    candidates = match_price(ball, prices, "fingerprint")
    assert [(candidate.source_row, candidate.product_name) for candidate in candidates] == [(8, "异径三通")]
    assert ball.product_name == "异径三通"
    assert required_categories(ball, prices) == {"球铁"}

    malleable = product(category="玛钢", product_name="异径三通", specification="25", price_marker="b")
    assert match_price(malleable, prices, "fingerprint") == []
    assert malleable.match_status == "not_found"


def test_required_categories_returns_all_ui_categories_regardless_of_existing_prefill():
    prices = [
        PriceRecord("m.xlsx", "沟槽数据", 6, "三通", "25", "b", Decimal("3.00"), 10, category="沟槽"),
        PriceRecord("m.xlsx", "玛钢数据", 7, "三通", "25", "b", Decimal("1.00"), 20, category="玛钢"),
        PriceRecord("m.xlsx", "玛钢数据", 8, "球热三通", "25", "b", Decimal("2.00"), 30, category="玛钢"),
    ]
    item = product(category="玛钢", product_name="三通", specification="25", price_marker="b")
    assert required_categories(item, prices) == {"沟槽", "玛钢", "球铁"}


@pytest.mark.parametrize("prefix", ["球热", "球铁"])
def test_explicit_ball_iron_name_limits_required_category_before_matching(prefix):
    prices = [
        PriceRecord("m.xlsx", "沟槽数据", 6, "三通", "25", "b", Decimal("3.00"), 10, category="沟槽"),
        PriceRecord("m.xlsx", "玛钢数据", 7, "三通", "25", "b", Decimal("1.00"), 20, category="玛钢"),
        PriceRecord("m.xlsx", "玛钢数据", 8, f"{prefix}三通", "25", "b", Decimal("2.00"), 30, category="玛钢"),
    ]
    recognized = ProductRecord(
        source_type="text", source_name="test", product_name=f"{prefix}三通",
        specification="25", quantity_cartons=1,
    )
    assert required_categories(recognized, prices) == {"球铁"}

    confirmed = product(category="球铁", product_name=f"{prefix}三通", specification="25", price_marker="b")
    assert [candidate.source_row for candidate in match_price(confirmed, prices, "fingerprint")] == [8]
    assert confirmed.product_name == f"{prefix}三通"


def test_prefixed_ball_iron_formal_name_is_not_hidden_by_same_base_malleable_alias():
    mappings = (("玛钢", "管古", ("管箍", "异径管箍")),)
    prices = [
        PriceRecord(
            "m.xlsx", "玛钢数据", 8, "球铁管古", "32", "b", Decimal("2.00"), 30,
            category="玛钢", multi_target_aliases=mappings,
        )
    ]
    recognized = ProductRecord(
        source_type="text", source_name="test", product_name="球铁管古",
        specification="32", quantity_cartons=2,
    )
    assert required_categories(recognized, prices) == {"球铁"}

    confirmed = product(category="球铁", product_name="球铁管古", specification="32", price_marker="b")
    assert [candidate.source_row for candidate in match_price(confirmed, prices, "fingerprint")] == [8]
    assert confirmed.product_name == "球铁管古"


def test_required_categories_keeps_category_scoped_aliases_isolated():
    mappings = (("玛钢", "玛钢专用T", ("三通",)),)
    prices = [
        PriceRecord(
            "m.xlsx", "沟槽数据", 6, "三通", "25", "b", Decimal("2.00"), 10,
            category="沟槽", multi_target_aliases=mappings,
        ),
        PriceRecord(
            "m.xlsx", "玛钢数据", 7, "三通", "25", "b", Decimal("1.00"), 20,
            category="玛钢", multi_target_aliases=mappings,
        ),
    ]
    item = product(category="沟槽", product_name="玛钢专用T", specification="25", price_marker="b")
    assert required_categories(item, prices) == {"玛钢"}


def test_malleable_multi_target_alias_cannot_match_groove_record():
    mappings = (("玛钢", "玛钢专用T", ("三通",)),)
    prices = [
        PriceRecord(
            "m.xlsx", "沟槽数据", 6, "三通", "25", "b", Decimal("2.00"), 10,
            category="沟槽", multi_target_aliases=mappings,
        ),
        PriceRecord(
            "m.xlsx", "玛钢数据", 7, "三通", "25", "b", Decimal("1.00"), 20,
            category="玛钢", multi_target_aliases=mappings,
        ),
    ]
    assert "玛钢专用t" not in price_aliases(prices, "沟槽")
    assert price_aliases(prices, "玛钢")["玛钢专用t"] == frozenset({"三通"})
    groove = product(category="沟槽", product_name="玛钢专用T", specification="25", price_marker="b")
    assert match_price(groove, prices, "fingerprint") == []
    assert groove.match_status == "not_found"
    malleable = product(category="玛钢", product_name="玛钢专用T", specification="25", price_marker="b")
    assert len(match_price(malleable, prices, "fingerprint")) == 1
    assert malleable.match_status == "matched"
    assert malleable.unit_price == Decimal("1.00")


def test_malleable_ordinary_alias_cannot_match_groove_but_matches_malleable():
    prices = [
        PriceRecord("m.xlsx", "沟槽数据", 6, "三通", "25", "b", Decimal("2.00"), 10, category="沟槽"),
        PriceRecord(
            "m.xlsx", "玛钢数据", 7, "三通", "25", "b", Decimal("1.00"), 20,
            aliases=("玛钢普通T",), category="玛钢",
        ),
    ]
    assert "玛钢普通t" not in price_aliases(prices, "沟槽")
    assert price_aliases(prices, "玛钢")["玛钢普通t"] == frozenset({"三通"})
    groove = product(category="沟槽", product_name="玛钢普通T", specification="25", price_marker="b")
    assert match_price(groove, prices, "fingerprint") == []
    assert groove.match_status == "not_found"
    malleable = product(category="玛钢", product_name="玛钢普通T", specification="25", price_marker="b")
    assert len(match_price(malleable, prices, "fingerprint")) == 1
    assert malleable.match_status == "matched"
    assert malleable.unit_price == Decimal("1.00")


def test_malleable_alias_does_not_hide_same_spelling_groove_formal_name():
    mappings = (("玛钢", "管古", ("管箍",)),)
    groove_price = PriceRecord(
        "m.xlsx", "沟槽数据", 6, "管古", "25", "b", Decimal("2.00"), 10,
        category="沟槽", multi_target_aliases=mappings,
    )
    item = product(category="沟槽", product_name="管古", specification="25", price_marker="b")
    assert len(match_price(item, [groove_price], "fingerprint")) == 1
    assert item.match_status == "matched"
    assert item.unit_price == Decimal("2.00")


def test_text_file_input():
    rows = extract_file("ask.txt", "165弯头200件".encode())
    assert len(rows) == 1
    assert rows[0].quantity_cartons == 200


def test_pdf_text_input():
    import fitz
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "165 elbow 200")
    assert extract_file("ask.pdf", document.tobytes()) == []


def test_scanned_pdf_falls_back_without_crashing():
    import fitz
    document = fitz.open()
    document.new_page()
    assert extract_file("scan.pdf", document.tobytes()) == []


class FakeVisionExtractor:
    def __init__(self):
        self.contexts = []

    def extract_products(self, image_bytes, context=None):
        self.contexts.append(context)
        return [
            product(
                source_type="image",
                source_name="temporary",
                source_location="区域 (1,2)-(3,4)",
                confidence=0.61,
                extraction_method="fake AI",
                manually_confirmed=False,
            )
        ]


class FakeTextExtractor:
    def __init__(self):
        self.texts = []

    def extract_products(self, text):
        self.texts.append(text)
        return [product(extraction_method="fake AI", manually_confirmed=False)]


def test_text_pdf_uses_extracted_text_before_vision():
    import fitz

    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "customer inquiry")
    text_ai, vision_ai = FakeTextExtractor(), FakeVisionExtractor()
    rows = extract_file("text.pdf", document.tobytes(), ai_text=text_ai, ai_vision=vision_ai)
    assert text_ai.texts == ["customer inquiry"]
    assert vision_ai.contexts == []
    assert rows[0].source_location == "第 1 页"


def test_pdf_with_non_product_text_falls_back_to_vision():
    import fitz

    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "header only")
    vision = FakeVisionExtractor()
    rows = extract_file("scan.pdf", document.tobytes(), ai_vision=vision)
    assert vision.contexts == ["scan.pdf 第 1 页"]
    assert rows[0].source_type == "pdf"
    assert rows[0].source_name == "scan.pdf"
    assert rows[0].source_location == "第 1 页 / 区域 (1,2)-(3,4)"
    assert AI_REVIEW_WARNING in rows[0].warnings


def test_image_input_without_ai(tmp_path):
    from PIL import Image
    output = BytesIO()
    Image.new("RGB", (20, 20), "white").save(output, "PNG")
    assert validate_image(output.getvalue()).startswith(b"\xff\xd8")
    assert extract_file("hand.png", output.getvalue()) == []


def test_image_routes_to_ai_vision_and_preserves_provenance():
    from PIL import Image

    output = BytesIO()
    Image.new("RGB", (20, 20), "white").save(output, "PNG")
    rows = extract_file("hand.png", output.getvalue(), ai_vision=FakeVisionExtractor())
    assert rows[0].source_name == "hand.png"
    assert rows[0].source_location == "整张图片 / 区域 (1,2)-(3,4)"
    assert rows[0].confidence == 0.61
    assert AI_REVIEW_WARNING in rows[0].warnings


def ai_payload(**changes):
    item = {
        "product_name_raw": "弯头",
        "specification_raw": "50×40",
        "quantity": 20,
        "grade_or_bag_raw": "中",
        "source_text": "50×40弯头 中 20件",
        "uncertain_fields": [],
    }
    item.update(changes)
    return {"items": [item]}


def test_ai_schema_preserves_raw_grade_without_mapping():
    row = _records_from_ai(ai_payload())[0]
    assert (row.product_name, row.specification, row.quantity_cartons) == ("弯头", "50×40", 20)
    assert row.original_grade_text == "中"
    assert row.price_marker is None
    assert row.manually_confirmed is False
    assert AI_REVIEW_WARNING in row.warnings


def test_ai_schema_uses_supplied_master_grade_mapping():
    row = _records_from_ai(ai_payload(), grade_markers={"中": "a"})[0]
    assert row.price_marker == "a"


def test_text_and_ai_preserve_explicit_uppercase_a_marker():
    text_row = extract_text("65弯头 A 10件")[0]
    ai_row = _records_from_ai(ai_payload(specification_raw="65", grade_or_bag_raw="A"))[0]
    assert (text_row.specification, text_row.original_grade_text, text_row.price_marker) == ("65", "A", "A")
    assert (ai_row.specification, ai_row.original_grade_text, ai_row.price_marker) == ("65", "A", "A")


def test_ai_schema_rejects_wrong_quantity_type():
    with pytest.raises(ValueError, match="quantity 类型错误"):
        _records_from_ai(ai_payload(quantity="20"))


def test_ai_null_fields_are_kept_for_manual_correction():
    row = _records_from_ai(
        ai_payload(
            product_name_raw=None,
            specification_raw=None,
            quantity=None,
            uncertain_fields=["product_name_raw", "specification_raw", "quantity"],
        )
    )[0]
    assert (row.product_name, row.specification, row.quantity_cartons) == (None, None, None)
    assert any("必须人工填写" in warning for warning in row.warnings)


def test_local_ocr_runs_rules_and_safe_candidate_ranking():
    class Result:
        json = {
            "res": {
                "parsing_res_list": [
                    {"block_label": "text", "block_content": "25占5件", "block_bbox": [0, 0, 100, 20], "block_order": 1}
                ]
            }
        }

    class Pipeline:
        def predict(self, _):
            return [Result()]

    from PIL import Image

    image = BytesIO()
    Image.new("RGB", (20, 20), "white").save(image, "JPEG")
    extractor = LocalOCRExtractor(MASTER_DATA_FILE)
    extractor._pipeline = Pipeline()
    row = extractor.extract_products(image.getvalue(), "sample.jpg")[0]
    assert (row.product_name, row.specification, row.quantity_cartons) == (None, "25", 5)
    assert row.price_marker is None
    assert AI_REVIEW_WARNING in row.warnings


def _local_ocr_rows(blocks):
    class Result:
        json = {"res": {"parsing_res_list": blocks}}

    class Pipeline:
        def predict(self, _):
            return [Result()]

    image = BytesIO()
    Image.new("RGB", (20, 20), "white").save(image, "JPEG")
    extractor = LocalOCRExtractor(MASTER_DATA_FILE)
    extractor._pipeline = Pipeline()
    return extractor.extract_products(image.getvalue(), "sample.jpg")


def test_local_ocr_l_zhong_title_maps_a_without_becoming_product():
    rows = _local_ocr_rows([
        {
            "block_label": "text",
            "block_content": "L中=65占20件\n40×25T 30件",
            "block_bbox": [0, 0, 400, 100],
            "block_order": 1,
        }
    ])
    assert len(rows) == 2
    assert rows[0].original_text == "L中=65占20件"
    assert all((row.original_grade_text, row.price_marker) == ("中", "a") for row in rows)
    assert all(row.product_name not in {"L中", "球铁"} for row in rows)


def test_local_ocr_row_grade_conflict_blocks_title_grade_override():
    row = _local_ocr_rows([
        {
            "block_label": "text",
            "block_content": "L中=25弯头b10件",
            "block_bbox": [0, 0, 400, 100],
            "block_order": 1,
        }
    ])[0]
    assert row.original_grade_text == "b"
    assert row.price_marker is None
    assert any("冲突" in warning for warning in row.warnings)


def test_local_ocr_preserves_explicit_uppercase_a_marker():
    row = _local_ocr_rows([
        {"block_label": "text", "block_content": "65弯头A 10件", "block_bbox": [0, 0, 400, 100], "block_order": 1}
    ])[0]
    assert (row.specification, row.original_grade_text, row.price_marker) == ("65", "A", "A")


def test_local_ocr_ball_iron_title_pre_scans_same_block_and_stays_in_spatial_column():
    rows = _local_ocr_rows([
        {"block_label": "text", "block_content": "25三通10件\n球铁", "block_bbox": [0, 0, 200, 100], "block_order": 1},
        {"block_label": "text", "block_content": "32三通5件", "block_bbox": [10, 120, 210, 180], "block_order": 2},
        {"block_label": "text", "block_content": "25弯头7件", "block_bbox": [1000, 0, 1200, 80], "block_order": 3},
        {"block_label": "text", "block_content": "32弯头8件", "block_bbox": [1010, 100, 1210, 180], "block_order": 4},
    ])
    assert len(rows) == 4
    assert [row.category for row in rows] == ["球铁", "球铁", None, None]
    assert all(row.product_name != "球铁" for row in rows)


@pytest.mark.parametrize("label", ["paragraph_title", "header"])
def test_local_ocr_standalone_ball_iron_title_binds_only_to_right_column(label):
    rows = _local_ocr_rows([
        {"block_label": label, "block_content": "球铁", "block_bbox": [1000, 0, 1200, 40], "block_order": 1},
        {"block_label": "text", "block_content": "25三通10件", "block_bbox": [0, 100, 200, 160], "block_order": 2},
        {"block_label": "text", "block_content": "25弯头7件", "block_bbox": [1010, 100, 1210, 160], "block_order": 3},
    ])
    assert {row.quantity_cartons: row.category for row in rows} == {10: None, 7: "球铁"}


def test_local_ocr_plain_header_does_not_clear_right_column_title():
    rows = _local_ocr_rows([
        {"block_label": "paragraph_title", "block_content": "球铁", "block_bbox": [1000, 0, 1200, 40], "block_order": 1},
        {"block_label": "header", "block_content": "报价单", "block_bbox": [1000, 50, 1200, 80], "block_order": 2},
        {"block_label": "text", "block_content": "25三通10件", "block_bbox": [0, 100, 200, 160], "block_order": 3},
        {"block_label": "text", "block_content": "25弯头7件", "block_bbox": [1010, 100, 1210, 160], "block_order": 4},
    ])
    assert {row.quantity_cartons: row.category for row in rows} == {10: None, 7: "球铁"}


def test_local_ocr_equal_center_blocks_keep_two_column_title_scope():
    rows = _local_ocr_rows([
        {"block_label": "text", "block_content": "球铁\n25三通10件", "block_bbox": [0, 0, 200, 80], "block_order": 1},
        {"block_label": "text", "block_content": "32三通5件", "block_bbox": [0, 100, 200, 160], "block_order": 2},
        {"block_label": "text", "block_content": "25弯头7件", "block_bbox": [1000, 0, 1200, 80], "block_order": 3},
        {"block_label": "text", "block_content": "32弯头8件", "block_bbox": [1000, 100, 1200, 160], "block_order": 4},
    ])
    assert {row.quantity_cartons: row.category for row in rows} == {
        10: "球铁", 5: "球铁", 7: None, 8: None,
    }


def test_local_ocr_ambiguous_category_title_invalidates_stale_context():
    rows = _local_ocr_rows([
        {"block_label": "paragraph_title", "block_content": "球铁", "block_bbox": [1000, 0, 1200, 40], "block_order": 1},
        {"block_label": "text", "block_content": "25三通10件", "block_bbox": [0, 100, 200, 160], "block_order": 2},
        {"block_label": "text", "block_content": "25弯头7件", "block_bbox": [1010, 100, 1210, 160], "block_order": 3},
        {"block_label": "paragraph_title", "block_content": "玛钢", "block_bbox": [100, 200, 1100, 240], "block_order": 4},
        {"block_label": "text", "block_content": "32三通5件", "block_bbox": [10, 300, 210, 360], "block_order": 5},
        {"block_label": "text", "block_content": "32弯头8件", "block_bbox": [1020, 300, 1220, 360], "block_order": 6},
    ])
    assert {row.quantity_cartons: row.category for row in rows} == {
        10: None, 7: "球铁", 5: None, 8: None,
    }


def test_local_ocr_ambiguous_standalone_title_does_not_propagate():
    rows = _local_ocr_rows([
        {"block_label": "paragraph_title", "block_content": "球铁", "block_bbox": [100, 0, 320, 40], "block_order": 1},
        {"block_label": "text", "block_content": "25三通10件", "block_bbox": [0, 100, 200, 160], "block_order": 2},
        {"block_label": "text", "block_content": "25弯头7件", "block_bbox": [220, 100, 420, 160], "block_order": 3},
    ])
    assert {row.quantity_cartons: row.category for row in rows} == {10: None, 7: None}


def test_local_ocr_ball_iron_title_does_not_cross_two_geometric_blocks():
    rows = _local_ocr_rows([
        {"block_label": "text", "block_content": "球铁\n25三通10件", "block_bbox": [0, 0, 200, 100], "block_order": 1},
        {"block_label": "text", "block_content": "25弯头7件", "block_bbox": [1000, 200, 1200, 280], "block_order": 2},
    ])
    assert {row.quantity_cartons: row.category for row in rows} == {10: "球铁", 7: None}


def test_local_ocr_ball_iron_title_does_not_cross_paired_slots():
    rows = _local_ocr_rows([
        {
            "block_label": "text",
            "block_content": "球铁\n25三通10件 25弯头7件",
            "block_bbox": [0, 0, 1200, 100],
            "block_order": 1,
        }
    ])
    assert {row.quantity_cartons: row.category for row in rows} == {10: "球铁", 7: None}


@pytest.mark.parametrize("grade,expected", [("a", "a"), ("A", "A"), ("b", "b"), ("B", "b")])
def test_local_ocr_preserves_explicit_letter_grade(grade, expected):
    class Result:
        json = {
            "res": {
                "parsing_res_list": [
                    {"block_label": "text", "block_content": f"25弯头{grade}10件", "block_bbox": [0, 0, 100, 20], "block_order": 1}
                ]
            }
        }

    class Pipeline:
        def predict(self, _):
            return [Result()]

    from PIL import Image

    image = BytesIO()
    Image.new("RGB", (20, 20), "white").save(image, "JPEG")
    extractor = LocalOCRExtractor(MASTER_DATA_FILE)
    extractor._pipeline = Pipeline()
    row = extractor.extract_products(image.getvalue(), "sample.jpg")[0]
    assert row.original_grade_text == grade
    assert row.price_marker == expected


def test_ai_unconfigured_degrades():
    assert extract_text("无法识别的自由文字") == []


def make_price_book(path: Path):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "标准数据"
    sheet.append(["记录键", "品名", "规格", "价格标记", "单价", "单重", "包装数"])
    sheet.append(["弯头|25|a", "弯头", "25", "a", 1.61, 126, 200])
    workbook.save(path)


MASTER_HEADERS = [
    "分类", "源代码", "商标", "品名", "规格", "别名", "价格标记", "轻重", "单价", "包装数",
    "毛坯单重", "成品单重", "纯毛单重", "启用状态", "记录键", "备注",
]


def make_master_book(path: Path, rows: list[list[object]]):
    workbook = Workbook()
    groove = workbook.active
    groove.title = "沟槽数据"
    malleable = workbook.create_sheet("玛钢数据")
    for sheet in (groove, malleable):
        sheet.append(["标准产品价格主数据"])
        sheet.append([])
        sheet.append([])
        sheet.append([])
        sheet.append(MASTER_HEADERS)
    for row in rows:
        if len(row) == 11:
            name, spec, aliases, marker, amount, packing, weight, _, status, _, note = row
            category = "玛钢"
            notes = str(note or "")
            if packing in (None, "") and "缺包装数" not in notes:
                notes = "；".join(filter(None, [notes, "缺包装数"]))
            row = [
                category, "test-code", "DEMO", name, spec, aliases, marker,
                {"a": "轻货", "A": "", "b": "重货"}.get(marker, "重货"), amount, packing,
                weight or 1, weight or 1, weight or 1, status,
                f"{category}｜{name}｜{normalize_specification(spec)}｜{marker}", notes,
            ]
        (groove if row[0] == "沟槽" else malleable).append(row)
    grades = workbook.create_sheet("等级映射")
    grades.append(["客户等级/袋色 → 价格标记"])
    grades.append([])
    grades.append([])
    grades.append(["客户写法", "价格标记", "启用状态", "备注"])
    grades.append(["蓝袋", "b", "启用", "测试"])
    grades["F13"] = "分类"
    grades["G13"] = "别名"
    grades["H13"] = "正式品名"
    grades["I13"] = "启用状态"
    grades["J13"] = "备注"
    grades.append([None, None, None, None, None, "玛钢", "测试多义", "目标一", "启用", "测试"])
    grades.append([None, None, None, None, None, "玛钢", "测试多义", "目标二", "启用", "测试"])
    workbook.save(path)


@pytest.fixture
def quote_environment(tmp_path, monkeypatch):
    master_dir = tmp_path / "主数据"
    output_dir = tmp_path / "data" / "output"
    master_dir.mkdir()
    output_dir.mkdir(parents=True)
    master_path = master_dir / "任意正式文件名.xlsx"
    make_master_book(master_path, [["弯头", "25", "", "a", 1.61, 200, "", "", "启用", "k1", ""]])
    monkeypatch.setattr("src.core.OUTPUT_ROOT", output_dir)
    monkeypatch.setattr("src.core.CUSTOMER_HISTORY_PATH", tmp_path / "data" / "settings" / "customer_names.json")
    return master_dir, output_dir, load_master_price_records(master_path), master_data_fingerprint(master_path)


@pytest.fixture
def dual_quote_environment(quote_environment):
    master_dir, output_dir, _, _ = quote_environment
    master_path = discover_master_data(master_dir)
    make_master_book(
        master_path,
        [
            ["弯头", "25", "", "a", 1.61, 200, "", "", "启用", "k1", ""],
            ["弯头", "25", "", "b", 1.82, 180, "", "", "启用", "k2", ""],
        ],
    )
    return master_dir, output_dir, load_master_price_records(master_path), master_data_fingerprint(master_path)


def test_price_table_reading(tmp_path):
    path = tmp_path / "price.xlsx"
    make_price_book(path)
    rows = load_price_records([path])
    assert len(rows) == 1
    assert (rows[0].price_marker, rows[0].unit_price, rows[0].packing_quantity) == ("a", Decimal("1.61"), 200)


def test_master_data_discovery_uses_only_root_non_temp_xlsx(tmp_path):
    official = tmp_path / "已改名.xlsx"
    official.touch()
    (tmp_path / "~$已改名.xlsx").touch()
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "ignored.xlsx").touch()
    assert discover_master_data(tmp_path) == official
    (tmp_path / "second.xlsx").touch()
    with pytest.raises(ValueError, match="必须且只能"):
        discover_master_data(tmp_path)


def test_master_data_atomic_load_checks_expected_fingerprint(tmp_path):
    path = tmp_path / "master.xlsx"
    make_master_book(path, [["弯头", "25", "", "a", 1.61, 200, "", "", "启用", "k1", ""]])
    fingerprint = master_data_fingerprint(path)
    prices, markers, actual = load_master_data(path, fingerprint)
    assert prices[0].unit_price == Decimal("1.61")
    assert markers == {"蓝袋": "b"}
    assert actual == fingerprint
    with pytest.raises(ValueError, match="已变化"):
        load_master_data(path, "stale")


def test_synthetic_master_preserves_pending_and_missing_packaging():
    prices = load_master_price_records(MASTER_DATA_FILE)
    assert len(prices) == 22
    assert load_master_grade_markers(MASTER_DATA_FILE) == {"蓝袋": "b", "好": "b", "黑袋": "a", "中": "a"}
    assert sum(price.unit_price is None for price in prices) == 2
    assert sum(price.packing_quantity is None for price in prices) == 1
    assert {price.price_marker for price in prices} == {"a", "A", "b"}
    assert all("虚构" in price.notes for price in prices)
    assert all("禁止报价" in price.notes for price in prices if price.status == "待补充")
    keys = [(price.category, price.product_name, normalize_specification(price.specification), price.price_marker) for price in prices]
    assert len(keys) == len(set(keys))


def test_synthetic_master_elbow_65_has_isolated_a_uppercase_a_and_b_tiers():
    prices = load_master_price_records(MASTER_DATA_FILE)
    expected = {
        "a": (Decimal("19"), 10, Decimal("109"), Decimal("99"), "65a"),
        "A": (Decimal("20"), 10, Decimal("110"), Decimal("100"), "65A"),
        "b": (Decimal("21"), 10, Decimal("111"), Decimal("101"), "65"),
    }
    for marker, facts in expected.items():
        item = product(product_name="弯头", specification="65", price_marker=marker)
        candidates = match_price(item, prices, "fingerprint")
        assert len(candidates) == 1
        assert (item.unit_price, item.packing_quantity, candidates[0].blank_weight, candidates[0].finished_weight, _sales_specification(item)) == facts


def test_blank_master_template_has_same_structure_but_is_not_runtime_source():
    template_path = Path("模板/标准产品价格表_空白模板.xlsx")
    workbook = load_workbook(template_path, data_only=True)
    assert workbook.sheetnames == ["沟槽数据", "玛钢数据", "等级映射"]
    for sheet_name in ("沟槽数据", "玛钢数据"):
        rows = list(workbook[sheet_name].iter_rows(min_row=6, values_only=True))
        assert not any(any(value not in (None, "") for value in row) for row in rows)
        assert any(
            validation.formula1 == '"a,A,b"' and "G6" in validation.cells
            for validation in workbook[sheet_name].data_validations.dataValidation
        )
    workbook.close()
    assert discover_master_data(Path("主数据")).resolve() == MASTER_DATA_FILE.resolve()


def test_master_data_preflight_blocks_invalid_sources(tmp_path):
    path = tmp_path / "标准产品价格表.xlsx"
    make_master_book(path, [])
    with pytest.raises(ValueError, match="没有可检索"):
        load_master_price_records(path)

    make_master_book(path, [["弯头", "25", "", "c", 1.61, 200, "", "", "启用", "k1", ""]])
    with pytest.raises(ValueError, match="价格标记"):
        load_master_price_records(path)

    make_master_book(path, [["弯头", "25", "", "A", 1.61, 200, "", "", "启用", "k1", "轻重含义待确认"]])
    assert load_master_price_records(path)[0].price_marker == "A"
    workbook = load_workbook(path)
    workbook["玛钢数据"]["H6"] = "轻货"
    workbook.save(path)
    workbook.close()
    with pytest.raises(ValueError, match="轻重必须与价格标记一致"):
        load_master_price_records(path)

    make_master_book(path, [["弯头", "25", "", "a", 1.61, 1.5, "", "", "启用", "k1", ""]])
    with pytest.raises(ValueError, match="包装数必须是正整数"):
        load_master_price_records(path)

    make_master_book(path, [["弯头", "25", "", "a", 1.61, 200, "", "", "启用", "k1", ""]])
    workbook = load_workbook(path)
    workbook["等级映射"]["B5"] = "c"
    workbook.save(path)
    workbook.close()
    with pytest.raises(ValueError, match="客户写法或价格标记非法"):
        load_master_price_records(path)

    make_master_book(path, [["弯头", "25", "", "a", 1.61, 200, "", "", "启用", "k1", ""]])
    workbook = load_workbook(path)
    workbook["等级映射"]["B5"] = "A"
    workbook.save(path)
    workbook.close()
    with pytest.raises(ValueError, match="客户写法或价格标记非法"):
        load_master_price_records(path)

    duplicate = ["弯头", "25", "", "a", 1.61, 200, "", "", "启用", "k1", ""]
    make_master_book(path, [duplicate, [*duplicate[:9], "k2", ""]])
    with pytest.raises(ValueError, match="业务键重复"):
        load_master_price_records(path)


def test_master_data_preflight_blocks_multiple_workbooks(tmp_path):
    path = tmp_path / "标准产品价格表.xlsx"
    row = ["弯头", "25", "刚性弯头", "a", 1.61, 200, "", "", "启用", "k1", ""]
    make_master_book(path, [row])
    make_master_book(tmp_path / "额外.xlsx", [row])
    with pytest.raises(ValueError, match="必须且只能包含"):
        load_master_price_records(path)


def test_apply_price_calculates_total():
    item = product()
    apply_price(item, price(), "fingerprint")
    assert item.total_price == Decimal("644.00")
    assert item.master_packing_quantity == 200


def test_unconfirmed_recognition_blocks_generation(quote_environment):
    master_dir, output_dir, _, _ = quote_environment
    with pytest.raises(ValueError, match="未确认"):
        generate_quotation([product(manually_confirmed=False)], output_dir, "测试单位", master_data_dir=master_dir)
    assert not list(output_dir.iterdir())


def test_missing_price_blocks_generation(quote_environment):
    master_dir, output_dir, _, _ = quote_environment
    item = product(manually_confirmed=True)
    with pytest.raises(ValueError, match="来源或内容指纹"):
        generate_quotation([item], output_dir, "测试单位", master_data_dir=master_dir)
    assert not list(output_dir.iterdir())


def test_basic_quotation_generation_and_reopen(quote_environment):
    master_dir, output_dir, prices, fingerprint = quote_environment
    item = product(price_marker="a")
    match_price(item, prices, fingerprint)
    path, mapping = generate_quotation(
        [item], output_dir, "  杭州   测试单位  ", master_data_dir=master_dir,
        generated_on=date(2026, 8, 29), sales_id="QA-ABCDE-23456",
    )
    assert path.name == "QA-ABCDE-23456.pdf"
    assert mapping["png_paths"] == [output_dir / "QA-ABCDE-23456_p1.png"]
    with fitz.open(path) as document:
        assert document.page_count == 1
        page = document[0]
        text = page.get_text().replace("\xa0", " ")
        words = page.get_text("words")
        spans = [
            span
            for block in page.get_text("dict")["blocks"]
            for line in block.get("lines", [])
            for span in line["spans"]
        ]
    assert all(value in text for value in (
        "示例企业销售单 - 虚构数据", "杭州 测试单位", "2026/8/29", "QA-ABCDE-23456", "90°弯头", "25a", "箱数",
        "零个", "下浮", "重量：", "644.00", "陆佰肆拾肆元整",
    ))
    assert all(value not in text for value in ("玛钢", "沟槽", "DEMO", "吨", "（", "）"))
    assert all(any(font in span["font"] for font in ("Microsoft YaHei", "Droid Sans Fallback")) for span in spans)
    assert next(span for span in spans if span["text"].replace("\xa0", " ") == "示例企业销售单 - 虚构数据")["size"] == 17
    assert next(span for span in spans if span["text"].startswith("共1页"))["size"] == 9
    assert next(span for span in spans if span["text"] == "日期：2026/8/29")["size"] == 10
    assert next(span for span in spans if span["text"] == "品名")["size"] == 10
    assert next(span for span in spans if span["text"] == "90°弯头")["size"] == 10
    assert next(span for span in spans if span["text"] == "负责人：")["size"] == 10
    sales_id_words = [word for word in words if word[4].endswith("QA-ABCDE-23456")]
    assert len(sales_id_words) == 1
    assert sales_id_words[0][2] > 690
    data_words = [word for word in words if 131 <= (word[1] + word[3]) / 2 <= 152]
    for value, expected_center in {
        "90°弯头": 123, "25a": 217, "箱": 280, "2": 319,
        "200": 361, "400": 450, "1.61": 507, "644.00": 571,
    }.items():
        word = next(word for word in data_words if word[4] == value)
        assert abs((word[0] + word[2]) / 2 - expected_center) < 1
    summary_words = [word for word in words if 152 <= (word[1] + word[3]) / 2 <= 175]
    summary_total = next(word for word in summary_words if word[4] == "合计金额")
    assert abs((summary_total[0] + summary_total[2]) / 2 - 571) < 1
    assert not any(28 <= (word[0] + word[2]) / 2 <= 73 for word in data_words)
    assert not any(383 <= (word[0] + word[2]) / 2 <= 421 for word in data_words)
    assert not any(607 <= (word[0] + word[2]) / 2 <= 694 for word in data_words)
    assert not any(479 <= (word[0] + word[2]) / 2 <= 535 and 152 <= (word[1] + word[3]) / 2 <= 175 for word in words)
    assert not any(607 <= (word[0] + word[2]) / 2 <= 694 and 175 <= (word[1] + word[3]) / 2 <= 201 for word in words)
    with Image.open(mapping["png_paths"][0]) as image:
        assert image.size == (1444, 788)
        image.verify()
    assert mapping["total_label"] == "完整总计"
    assert mapping["total"] == Decimal("644.00")
    assert load_customer_names() == ["杭州 测试单位"]
    records = list((output_dir.parent / "settings" / "quotation_records").rglob("*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text(encoding="utf-8"))
    assert record == {
        "sales_id": "QA-ABCDE-23456",
        "document_date": "2026-08-29",
        "created_at": record["created_at"],
        "customer_name": "杭州 测试单位",
        "output_files": ["QA-ABCDE-23456.pdf", "QA-ABCDE-23456_p1.png"],
        "category_keys": [],
    }
    created_at = datetime.fromisoformat(record["created_at"])
    assert created_at.utcoffset().total_seconds() == 8 * 60 * 60
    assert "玛钢" not in records[0].read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "specification,marker,expected",
    [("25", "a", "25a"), ("25a", "a", "25a"), ("25", "A", "25A"), ("25A", "A", "25A"), ("25", "b", "25")],
)
def test_sales_slip_specification_preserves_marker_case(specification, marker, expected):
    assert _sales_specification(product(specification=specification, price_marker=marker)) == expected


def test_render_quotation_preview_returns_png_bytes_without_writes_or_persistence(quote_environment, monkeypatch):
    master_dir, output_dir, _, _ = quote_environment
    master_path = discover_master_data(master_dir)
    make_master_book(
        master_path,
        [
            ["弯头", "25", "", "a", 1.61, 200, "", "", "启用", "k1", ""],
            ["待补充产品", "25", "", "b", None, 20, "", "", "待补充", "k2", "禁止报价"],
        ],
    )
    prices = load_master_price_records(master_path)
    fingerprint = master_data_fingerprint(master_path)
    matched = product(price_marker="a")
    pending = product(product_name="待补充产品", price_marker="b")
    match_price(matched, prices, fingerprint)
    match_price(pending, prices, fingerprint)
    rendered: dict[str, object] = {}
    from src import core

    real_renderer = core._render_quote_files

    def capture_renderer(products, customer_name, generated_on, sales_id, incomplete_total):
        rendered.update(products=products, customer_name=customer_name, sales_id=sales_id)
        return real_renderer(products, customer_name, generated_on, sales_id, incomplete_total)

    monkeypatch.setattr(core, "_render_quote_files", capture_renderer)
    monkeypatch.setattr(core, "generate_sales_id", lambda: pytest.fail("preview allocated a sales id"))
    monkeypatch.setattr(core, "remember_customer_name", lambda *_args, **_kwargs: pytest.fail("preview persisted history"))
    before = set(master_dir.parent.rglob("*"))
    png_bytes = render_quotation_preview(
        [matched, pending], "  预览单位  ", master_data_dir=master_dir, generated_on=date(2026, 9, 3)
    )

    assert rendered == {"products": [matched], "customer_name": "预览单位", "sales_id": "PREVIEW"}
    assert set(master_dir.parent.rglob("*")) == before
    assert len(png_bytes) == 1 and png_bytes[0].startswith(b"\x89PNG\r\n\x1a\n")
    with Image.open(BytesIO(png_bytes[0])) as image:
        assert image.size == (1444, 788)
        image.verify()
    with pytest.raises(ValueError, match="没有可生成的有价产品"):
        render_quotation_preview([pending], "预览单位", master_data_dir=master_dir, generated_on=date(2026, 9, 3))


def test_render_quotation_preview_blocks_stale_master_without_writing(quote_environment):
    master_dir, output_dir, prices, fingerprint = quote_environment
    item = product(price_marker="a")
    match_price(item, prices, fingerprint)
    master_path = discover_master_data(master_dir)
    workbook = load_workbook(master_path)
    workbook["玛钢数据"]["I6"] = 1.62
    workbook.save(master_path)
    workbook.close()

    with pytest.raises(ValueError, match="旧价格已失效"):
        render_quotation_preview([item], "预览单位", master_data_dir=master_dir)
    assert not list(output_dir.iterdir())
    assert load_customer_names() == []


def test_sales_id_is_private_ambiguity_reduced_base32():
    identifiers = {generate_sales_id() for _ in range(32)}
    assert len(identifiers) == 32
    assert all(re.fullmatch(r"QA-[23456789ABCDEFGHJKLMNPQRSTUVWXYZ]{5}-[23456789ABCDEFGHJKLMNPQRSTUVWXYZ]{5}", value) for value in identifiers)
    assert all(not any(token in value[3:] for token in ("2026", "玛钢", "测试单位", "_", "0", "O", "1", "I")) for value in identifiers)


@pytest.mark.parametrize("sales_id", ["QA-ABCDE-23450", "QA-ABCDE-2345O", "QA-ABCDE-23451", "QA-ABCDE-2345I"])
def test_generation_rejects_ambiguous_manual_sales_id(quote_environment, sales_id):
    master_dir, output_dir, prices, fingerprint = quote_environment
    item = product(price_marker="a")
    match_price(item, prices, fingerprint)
    with pytest.raises(ValueError, match="去歧义大写 Base32"):
        generate_quotation([item], output_dir, "测试单位", master_data_dir=master_dir, sales_id=sales_id)
    assert not list(output_dir.iterdir())
    assert load_customer_names() == []


def test_chinese_money_blocks_unsupported_large_amount():
    with pytest.raises(ValueError, match="金额过大"):
        _chinese_money(Decimal("10000000000000000"))


def test_generation_date_uses_shanghai_business_timezone(monkeypatch):
    real_datetime = __import__("datetime").datetime

    class FixedDateTime:
        @classmethod
        def now(cls, timezone):
            assert timezone.key == "Asia/Shanghai"
            return real_datetime(2026, 8, 29, 0, 1, tzinfo=timezone)

    monkeypatch.setattr("src.core.datetime", FixedDateTime)
    assert local_generation_date() == date(2026, 8, 29)


def test_customer_history_is_normalized_deduplicated_and_reloads(tmp_path):
    path = tmp_path / "settings" / "customers.json"
    remember_customer_name("  杭州   测试单位  ", path)
    remember_customer_name("杭州 测试单位", path)
    assert load_customer_names(path) == ["杭州 测试单位"]


def test_customer_history_rejects_non_string_elements(tmp_path):
    path = tmp_path / "settings" / "customers.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(["合法单位", 123], ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="每一项都必须是字符串"):
        load_customer_names(path)


def test_customer_history_keeps_two_concurrent_session_names(tmp_path):
    path = tmp_path / "settings" / "customers.json"
    names = ["并发单位甲", "并发单位乙"]
    with ThreadPoolExecutor(max_workers=2) as executor:
        assert list(executor.map(lambda name: remember_customer_name(name, path), names)) == names
    assert set(load_customer_names(path)) == set(names)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("product_name", "不存在产品", "确认后已修改"),
        ("specification", "999", "确认后已修改"),
        ("unit_price", Decimal("99.99"), "不一致"),
        ("master_packing_quantity", 999, "不一致"),
        ("price_source_file", "forged.xlsx", "不一致"),
        ("price_source_sheet", "伪造工作表", "不一致"),
        ("price_source_row", 999, "不一致"),
    ],
)
def test_generation_rejects_forged_master_match_fields(quote_environment, field, value, error):
    master_dir, output_dir, prices, fingerprint = quote_environment
    item = product(price_marker="a")
    match_price(item, prices, fingerprint)
    setattr(item, field, value)
    with pytest.raises(ValueError, match=error):
        generate_quotation([item], output_dir, "测试单位", master_data_dir=master_dir)
    assert not list(output_dir.iterdir())
    assert load_customer_names() == []


def test_generation_rejects_coordinated_switch_to_other_legal_ab_record(dual_quote_environment):
    master_dir, output_dir, prices, fingerprint = dual_quote_environment
    item = product(price_marker="a")
    match_price(item, prices, fingerprint)
    item.price_marker = "b"
    apply_price(item, next(price for price in prices if price.price_marker == "b"), fingerprint)
    with pytest.raises(ValueError, match="确认后已修改"):
        generate_quotation([item], output_dir, "测试单位", master_data_dir=master_dir)


def test_price_marker_change_requires_reconfirmation_before_new_match(dual_quote_environment):
    _, _, prices, fingerprint = dual_quote_environment
    item = product(price_marker="a")
    match_price(item, prices, fingerprint)
    item.price_marker = "b"
    assert match_price(item, prices, fingerprint) == []
    assert item.match_status == "unconfirmed"
    assert item.unit_price is None
    confirm_inquiry(item)
    match_price(item, prices, fingerprint)
    assert item.match_status == "matched"
    assert item.unit_price == Decimal("1.82")
    assert item.confirmed_inquiry == ("玛钢", "90°弯头", "25", 2, "b")


@pytest.mark.parametrize(
    "field,value",
    [("category", "沟槽"), ("product_name", "三通"), ("specification", "32"), ("quantity_cartons", 3)],
)
def test_any_confirmed_inquiry_field_change_invalidates_match(field, value):
    item = product(price_marker="a")
    setattr(item, field, value)
    assert match_price(item, [price()], "fingerprint") == []
    assert item.match_status == "unconfirmed"


def test_preview_and_generation_reject_direct_packing_tamper(quote_environment):
    master_dir, output_dir, prices, fingerprint = quote_environment
    item = product(price_marker="a")
    match_price(item, prices, fingerprint)
    item.packing_quantity = 150
    with pytest.raises(ValueError, match="包装或价格来源与当前主数据不一致"):
        render_quotation_preview([item], "测试单位", master_data_dir=master_dir)
    with pytest.raises(ValueError, match="包装或价格来源与当前主数据不一致"):
        generate_quotation([item], output_dir, "测试单位", master_data_dir=master_dir)
    assert not list(output_dir.iterdir())
    assert load_customer_names() == []


def test_missing_packing_generates_blank_line_total_and_subtotal(quote_environment):
    master_dir, output_dir, _, _ = quote_environment
    master_path = discover_master_data(master_dir)
    make_master_book(master_path, [["弯头", "25", "", "a", 1.61, None, "", "", "启用", "k1", ""]])
    prices = load_master_price_records(master_path)
    fingerprint = master_data_fingerprint(master_path)
    item = product(price_marker="a")
    match_price(item, prices, fingerprint)
    with pytest.raises(ValueError, match="主数据无包装数，不得填写本单包装数"):
        apply_packing_override(item, 200)
    assert item.packing_quantity is None
    assert item.total_price is None
    item.packing_quantity = 200
    with pytest.raises(ValueError, match="包装或价格来源与当前主数据不一致"):
        render_quotation_preview([item], "测试单位", master_data_dir=master_dir)
    with pytest.raises(ValueError, match="包装或价格来源与当前主数据不一致"):
        generate_quotation([item], output_dir, "测试单位", master_data_dir=master_dir)
    assert not list(output_dir.iterdir())
    assert load_customer_names() == []
    item.packing_quantity = None
    path, mapping = generate_quotation([item], output_dir, "测试单位", master_data_dir=master_dir)
    with fitz.open(path) as document:
        page = document[0]
        text = page.get_text()
        row_words = [word for word in page.get_text("words") if 131 <= (word[1] + word[3]) / 2 <= 152]
    assert "已计算小计" in text
    assert "完整总计" not in text
    assert all(value not in [word[4] for word in row_words] for value in ("0.00", "200", "400"))
    assert not any(
        left <= (word[0] + word[2]) / 2 <= right
        for word in row_words
        for left, right in ((339, 383), (421, 479), (535, 607))
    )
    assert mapping["total_label"] == "已计算小计"
    assert mapping["total"] == Decimal("0.00")


def test_pending_price_is_skipped_without_blocking_normal_export(quote_environment):
    master_dir, output_dir, _, _ = quote_environment
    master_path = discover_master_data(master_dir)
    make_master_book(
        master_path,
        [
            ["弯头", "25", "", "a", 1.61, 200, "", "", "启用", "k1", ""],
            ["90°弯头", "114 22.5", "", "b", None, 20, "", "", "待补充", "k2", "禁止报价"],
        ],
    )
    prices = load_master_price_records(master_path)
    fingerprint = master_data_fingerprint(master_path)
    normal = product(price_marker="a")
    pending = product(product_name="弯头", specification="114 22.5", price_marker="b")
    match_price(normal, prices, fingerprint)
    match_price(pending, prices, fingerprint)
    assert pending.match_status == "pending"
    assert pending.unit_price is None
    assert pending.total_price is None

    path, mapping = generate_quotation([normal, pending], output_dir, "测试单位", master_data_dir=master_dir)
    with fitz.open(path) as document:
        text = "\n".join(page.get_text() for page in document)
    assert mapping["total"] == normal.total_price
    assert "25" in text
    assert "114 22.5" not in text

    with pytest.raises(ValueError, match="没有可生成的有价产品"):
        generate_quotation([pending], output_dir, "另一单位", master_data_dir=master_dir)


def test_generation_sums_already_ceiled_line_totals(quote_environment):
    master_dir, output_dir, _, _ = quote_environment
    master_path = discover_master_data(master_dir)
    make_master_book(master_path, [["弯头", "25", "", "a", 0.001, 1, "", "", "启用", "k1", ""]])
    prices = load_master_price_records(master_path)
    fingerprint = master_data_fingerprint(master_path)
    items = [product(quantity_cartons=1, price_marker="a") for _ in range(2)]
    for item in items:
        match_price(item, prices, fingerprint)
    path, mapping = generate_quotation(items, output_dir, "测试单位", master_data_dir=master_dir)
    with fitz.open(path) as document:
        text = document[0].get_text()
    assert "0.001" in text
    assert "0.00" not in text.split()
    assert [item.total_price for item in items] == [Decimal("0.01"), Decimal("0.01")]
    assert mapping["total"] == Decimal("0.02")


def test_unit_price_display_pads_to_two_decimals_without_changing_decimal(quote_environment):
    master_dir, output_dir, _, _ = quote_environment
    master_path = discover_master_data(master_dir)
    make_master_book(master_path, [["弯头", "25", "", "a", 11.8, 1, "", "", "启用", "k1", ""]])
    prices = load_master_price_records(master_path)
    fingerprint = master_data_fingerprint(master_path)
    item = product(quantity_cartons=1, price_marker="a")
    match_price(item, prices, fingerprint)
    path, mapping = generate_quotation([item], output_dir, "测试单位", master_data_dir=master_dir)
    with fitz.open(path) as document:
        text = document[0].get_text()
    assert item.unit_price == Decimal("11.8")
    assert "11.80" in text
    assert mapping["total"] == Decimal("11.80")


def test_seven_rows_fit_one_page_with_totals(quote_environment):
    master_dir, output_dir, _, _ = quote_environment
    master_path = discover_master_data(master_dir)
    make_master_book(
        master_path,
        [["弯头", f"DN00{index}", "", "a", 1.61, 200, "", "", "启用", f"k{index}", ""] for index in range(1, 8)],
    )
    prices = load_master_price_records(master_path)
    fingerprint = master_data_fingerprint(master_path)
    items = [product(specification=f"DN00{index}", price_marker="a") for index in range(1, 8)]
    for item in items:
        match_price(item, prices, fingerprint)
    path, mapping = generate_quotation(items, output_dir, "分页测试单位", master_data_dir=master_dir)
    with fitz.open(path) as document:
        assert document.page_count == 1
        text = document[0].get_text()
    assert all(f"DN00{index}" in text for index in range(1, 8))
    assert all(text.count(f"DN00{index}") == 1 for index in range(1, 8))
    assert all(label in text for label in ("合计金额", "人民币大写", "负责人："))
    assert mapping["page_count"] == 1
    assert len(mapping["png_paths"]) == 1


def test_eight_rows_preserve_order_and_only_last_page_has_totals(quote_environment):
    master_dir, output_dir, _, _ = quote_environment
    master_path = discover_master_data(master_dir)
    make_master_book(
        master_path,
        [["弯头", f"DN00{index}", "", "a", 1.61, 200, "", "", "启用", f"k{index}", ""] for index in range(1, 9)],
    )
    prices = load_master_price_records(master_path)
    fingerprint = master_data_fingerprint(master_path)
    items = [product(specification=f"DN00{index}", price_marker="a") for index in range(1, 9)]
    for item in items:
        match_price(item, prices, fingerprint)
    path, mapping = generate_quotation(items, output_dir, "分页测试单位", master_data_dir=master_dir)
    with fitz.open(path) as document:
        assert document.page_count == 2
        texts = [page.get_text() for page in document]
    assert all(f"DN00{index}" in texts[0] for index in range(1, 8))
    assert "DN008" not in texts[0]
    assert "DN008" in texts[1]
    combined_text = "\n".join(texts)
    assert all(combined_text.count(f"DN00{index}") == 1 for index in range(1, 9))
    assert all(label not in texts[0] for label in ("合计金额", "人民币大写", "负责人："))
    assert all(label in texts[1] for label in ("合计金额", "人民币大写", "负责人："))
    assert mapping["page_count"] == 2
    assert len(mapping["png_paths"]) == 2


def test_packing_override_rejects_value_different_from_master():
    item = product(price_marker="a")
    apply_price(item, price(), "fingerprint")
    with pytest.raises(ValueError, match="必须与主数据包装数一致"):
        apply_packing_override(item, 150)
    assert item.master_packing_quantity == 200
    assert item.packing_quantity == 200
    assert item.total_price == Decimal("644.00")
    apply_packing_override(item, 200)
    assert item.total_price == Decimal("644.00")
    with pytest.raises(ValueError, match="正整数"):
        apply_packing_override(item, 1.5)


def test_generation_rejects_stale_fingerprint(quote_environment):
    master_dir, output_dir, prices, fingerprint = quote_environment
    item = product(price_marker="a")
    match_price(item, prices, fingerprint)
    master_path = discover_master_data(master_dir)
    workbook = load_workbook(master_path)
    workbook["玛钢数据"]["I6"] = 1.62
    workbook.save(master_path)
    workbook.close()
    with pytest.raises(ValueError, match="旧价格已失效"):
        generate_quotation([item], output_dir, "测试单位", master_data_dir=master_dir)
    assert not list(output_dir.iterdir())
    assert load_customer_names() == []


def test_generation_rejects_existing_or_outside_target(quote_environment, tmp_path):
    master_dir, output_dir, prices, fingerprint = quote_environment
    item = product(price_marker="a")
    match_price(item, prices, fingerprint)
    sales_id = "QA-ABCDE-23456"
    existing = output_dir / f"{sales_id}_p1.png"
    existing.write_bytes(b"keep")
    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        generate_quotation(
            [item], output_dir, "测试单位", master_data_dir=master_dir,
            generated_on=date(2026, 8, 29), sales_id=sales_id,
        )
    assert existing.read_bytes() == b"keep"
    assert not (output_dir / f"{sales_id}.pdf").exists()
    assert load_customer_names() == []
    with pytest.raises(ValueError, match="data/output"):
        generate_quotation([item], tmp_path / "outside", "测试单位", master_data_dir=master_dir)


def test_generation_rejects_existing_private_record_without_writing_outputs(quote_environment, monkeypatch):
    master_dir, output_dir, prices, fingerprint = quote_environment
    item = product(price_marker="a")
    match_price(item, prices, fingerprint)
    record_path = output_dir.parent / "settings" / "quotation_records" / "existing.json"
    record_path.parent.mkdir(parents=True)
    record_path.write_bytes(b"keep")
    monkeypatch.setattr("src.core._quotation_record_path", lambda *_args: record_path)
    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        generate_quotation(
            [item], output_dir, "测试单位", master_data_dir=master_dir,
            sales_id="QA-ABCDE-23456",
        )
    assert record_path.read_bytes() == b"keep"
    assert not list(output_dir.iterdir())
    assert load_customer_names() == []


def test_generation_write_failure_rolls_back_outputs_and_history(quote_environment, monkeypatch):
    master_dir, output_dir, prices, fingerprint = quote_environment
    item = product(price_marker="a")
    match_price(item, prices, fingerprint)
    original_open = Path.open

    def failing_open(path, *args, **kwargs):
        if path.suffix == ".png" and args and args[0] == "xb":
            raise OSError("simulated write failure")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing_open)
    with pytest.raises(OSError, match="simulated"):
        generate_quotation([item], output_dir, "失败单位", master_data_dir=master_dir)
    assert not list(output_dir.iterdir())
    assert load_customer_names() == []


def test_history_write_failure_rolls_back_all_new_outputs(quote_environment, monkeypatch):
    master_dir, output_dir, prices, fingerprint = quote_environment
    item = product(price_marker="a")
    match_price(item, prices, fingerprint)

    def fail_history(*_args, **_kwargs):
        raise OSError("simulated history failure")

    monkeypatch.setattr("src.core.remember_customer_name", fail_history)
    with pytest.raises(OSError, match="history failure"):
        generate_quotation([item], output_dir, "失败单位", master_data_dir=master_dir)
    assert not list(output_dir.iterdir())
    assert load_customer_names() == []
    assert not list((output_dir.parent / "settings" / "quotation_records").rglob("*.json"))


def test_quotation_record_write_failure_rolls_back_outputs_and_history(quote_environment, monkeypatch):
    master_dir, output_dir, prices, fingerprint = quote_environment
    item = product(price_marker="a")
    match_price(item, prices, fingerprint)
    original_open = Path.open

    def failing_open(path, *args, **kwargs):
        if path.suffix == ".json" and args and args[0] == "xb":
            raise OSError("simulated record failure")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing_open)
    with pytest.raises(OSError, match="record failure"):
        generate_quotation([item], output_dir, "失败单位", master_data_dir=master_dir)
    assert not list(output_dir.iterdir())
    assert load_customer_names() == []
    assert not list((output_dir.parent / "settings" / "quotation_records").rglob("*.json"))


def test_quotation_record_readback_failure_rolls_back_outputs_and_history(quote_environment, monkeypatch):
    master_dir, output_dir, prices, fingerprint = quote_environment
    item = product(price_marker="a")
    match_price(item, prices, fingerprint)
    original_read_text = Path.read_text

    def corrupt_record_readback(path, *args, **kwargs):
        if path.parent.name.isdigit() and path.suffix == ".json":
            return "{}"
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", corrupt_record_readback)
    with pytest.raises(ValueError, match="后台记录回读不一致"):
        generate_quotation([item], output_dir, "失败单位", master_data_dir=master_dir)
    assert not list(output_dir.iterdir())
    assert load_customer_names() == []
    assert not list((output_dir.parent / "settings" / "quotation_records").rglob("*.json"))


def test_master_change_during_output_readback_rolls_back_files_and_history(quote_environment, monkeypatch):
    master_dir, output_dir, prices, fingerprint = quote_environment
    item = product(price_marker="a")
    match_price(item, prices, fingerprint)
    master_path = discover_master_data(master_dir)
    original_image_open = Image.open
    readback_count = 0

    def mutate_master_on_disk_readback(source, *args, **kwargs):
        nonlocal readback_count
        image = original_image_open(source, *args, **kwargs)
        if isinstance(source, Path) and source.suffix == ".png":
            readback_count += 1
            workbook = load_workbook(master_path)
            workbook["玛钢数据"]["I6"] = 1.62
            workbook.save(master_path)
            workbook.close()
        return image

    monkeypatch.setattr("src.core.Image.open", mutate_master_on_disk_readback)
    with pytest.raises(ValueError, match="生成过程中发生变化"):
        generate_quotation([item], output_dir, "测试单位", master_data_dir=master_dir)
    assert readback_count == 1
    assert not list(output_dir.iterdir())
    assert load_customer_names() == []
    assert not list((output_dir.parent / "settings" / "quotation_records").rglob("*.json"))


def test_generation_blocks_overflow_without_output_or_history(quote_environment):
    master_dir, output_dir, prices, fingerprint = quote_environment
    item = product(price_marker="a")
    match_price(item, prices, fingerprint)
    with pytest.raises(ValueError, match="字段过长"):
        generate_quotation([item], output_dir, "超长单位名称" * 100, master_data_dir=master_dir)
    assert not list(output_dir.iterdir())
    assert load_customer_names() == []


def test_corrupt_history_blocks_and_rolls_back_new_outputs(quote_environment):
    master_dir, output_dir, prices, fingerprint = quote_environment
    item = product(price_marker="a")
    match_price(item, prices, fingerprint)
    history_path = output_dir.parent / "settings" / "customer_names.json"
    history_path.parent.mkdir(parents=True)
    history_path.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="历史无法读取"):
        generate_quotation([item], output_dir, "测试单位", master_data_dir=master_dir)
    assert not list(output_dir.iterdir())
    assert history_path.read_text(encoding="utf-8") == "{broken"
    assert not list((output_dir.parent / "settings" / "quotation_records").rglob("*.json"))


def test_source_price_file_not_modified(tmp_path):
    path = tmp_path / "price.xlsx"
    make_price_book(path)
    original = path.read_bytes()
    load_price_records([path])
    assert path.read_bytes() == original
