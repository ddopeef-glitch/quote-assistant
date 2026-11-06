import copy
import json
from pathlib import Path

import pytest
from openpyxl import load_workbook

from src.agent_catalog import CATALOG_RECORD_FIELDS, CatalogSnapshot, validate_proposal
from scripts.create_demo_data import make_book


ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "主数据" / "标准产品价格表.xlsx"
FROZEN_CATALOG = ROOT / "tests" / "fixtures" / "demo_catalog.json"


@pytest.fixture(scope="module")
def live_catalog():
    return CatalogSnapshot.from_master(MASTER)


@pytest.fixture(scope="module")
def frozen_payload():
    return json.loads(FROZEN_CATALOG.read_text(encoding="utf-8"))


def test_live_catalog_keeps_frozen_products_and_adds_authorized_grade_equivalents(live_catalog, frozen_payload):
    frozen = CatalogSnapshot.from_eval_catalog(frozen_payload)
    expected = copy.deepcopy(frozen_payload)
    expected["grade_mappings"].extend([{"raw": "黑", "price_marker": "a"}, {"raw": "蓝", "price_marker": "b"}])
    expected["grade_mappings"].sort(key=lambda item: (item["raw"], item["price_marker"]))
    assert len(live_catalog.records) == 31
    assert business_projection(live_catalog.to_payload()) == business_projection(expected)
    proof = CatalogSnapshot.from_eval_catalog(live_catalog.to_payload()).assert_equivalent(live_catalog)
    assert proof["status"] == "verified"
    assert proof["record_count"] == 31
    assert proof["master_fingerprint"] == live_catalog.master_fingerprint
    assert frozen.catalog_fingerprint == "a335c2e3e5d9070884c8f722d57aae7a1a9872f58bea0afed31b91542cfb209c"


def business_projection(payload):
    projected = copy.deepcopy(payload)
    for product in projected["products"]:
        del product["catalog_id"]
    return projected


def test_regenerated_workbooks_keep_business_projection_but_invalidate_old_ids(tmp_path):
    snapshots = []
    for name in ("first", "second"):
        folder = tmp_path / name
        folder.mkdir()
        path = folder / "master.xlsx"
        make_book(path)
        with_metadata = load_workbook(path)
        with_metadata.properties.description = name
        with_metadata.save(path)
        with_metadata.close()
        snapshots.append(CatalogSnapshot.from_master(path))
    first, second = snapshots
    assert business_projection(first.to_payload()) == business_projection(second.to_payload())
    assert first.master_fingerprint != second.master_fingerprint
    first_ids = {item["catalog_id"] for item in first.to_payload()["products"]}
    second_ids = {item["catalog_id"] for item in second.to_payload()["products"]}
    assert len(first_ids) == len(second_ids) == 31
    assert first_ids.isdisjoint(second_ids)
    with pytest.raises(ValueError, match="不一致"):
        CatalogSnapshot.from_eval_catalog(first.to_payload()).assert_equivalent(second)


@pytest.mark.parametrize("separator", ["x", "×", "*", "-"])
def test_catalog_normalizes_supported_master_specifications(tmp_path, separator):
    from src.core import ProductRecord, confirm_inquiry, load_master_data, match_price

    path = tmp_path / "master.xlsx"
    make_book(path)
    workbook = load_workbook(path)
    for row in workbook["玛钢数据"]:
        for cell in row:
            if cell.value == "32x25x25":
                cell.value = separator.join(("32", "25", "25"))
    workbook.save(path)
    workbook.close()
    catalog = CatalogSnapshot.from_master(path)
    assert any(item["specification"] == "32x25x25" for item in catalog.to_payload()["products"])
    prices, _, fingerprint = load_master_data(path)
    item = ProductRecord("manual", "test", category="玛钢", product_name="三通",
                         specification="32x25x25", quantity_cartons=1, price_marker="a")
    confirm_inquiry(item)
    assert len(match_price(item, prices, fingerprint)) == 1


def test_master_change_during_snapshot_construction_is_rejected(monkeypatch):
    monkeypatch.setattr("src.agent_catalog.master_data_fingerprint", lambda _path: "changed")

    with pytest.raises(ValueError, match="构造期间发生变化"):
        CatalogSnapshot.from_master(MASTER)


def test_only_live_snapshot_can_recheck_current_master(live_catalog, frozen_payload):
    frozen = CatalogSnapshot.from_eval_catalog(frozen_payload)

    assert live_catalog.assert_current() == live_catalog.master_fingerprint
    with pytest.raises(ValueError, match="未绑定"):
        frozen.assert_current()


def test_frozen_catalog_is_only_self_consistent_until_compared_with_live_master(frozen_payload):
    one = CatalogSnapshot.from_eval_catalog(frozen_payload)
    two = CatalogSnapshot.from_eval_catalog(copy.deepcopy(frozen_payload))

    assert one.master_verified is False
    with pytest.raises(ValueError, match="不能证明主表对应关系"):
        one.assert_equivalent(two)


def test_index_has_no_specification_or_private_master_fields(live_catalog):
    index = live_catalog.index_payload()
    encoded = json.dumps(index, ensure_ascii=False)

    assert len(index["products"]) < len(live_catalog.records)
    assert "specification" not in encoded
    for forbidden in ("unit_price", "packing", "weight", "source_row", "source_file", "notes"):
        assert forbidden not in encoded


def test_query_preserves_alias_multi_targets_categories_markers_pending_and_all_matches(live_catalog):
    response = live_catalog.query_batch(
        [
            {
                "query_id": "multi-alias",
                "product_terms": ["三通"],
                "specifications": ["50*20"],
                "categories": ["玛钢"],
                "price_markers": [],
            },
            {
                "query_id": "bare-elbow",
                "product_terms": ["弯头"],
                "specifications": ["65"],
                "categories": ["玛钢"],
                "price_markers": [],
            },
            {
                "query_id": "pending",
                "product_terms": ["90弯头"],
                "specifications": ["11422.5"],
                "categories": ["沟槽"],
                "price_markers": ["b"],
            },
        ]
    )
    by_query = {result["query_id"]: result for result in response["results"]}

    multi = by_query["multi-alias"]
    assert multi["returned_count"] == multi["total_matches"] > 1
    assert len({record["product_name"] for record in multi["candidates"]}) > 1
    assert {record["category"] for record in multi["candidates"]} == {"玛钢"}
    assert all(set(record) == CATALOG_RECORD_FIELDS for record in multi["candidates"])

    elbows = by_query["bare-elbow"]["candidates"]
    assert {record["price_marker"] for record in elbows} == {"a", "A", "b"}
    assert {record["product_name"] for record in elbows} == {"弯头"}

    pending = by_query["pending"]["candidates"]
    assert len(pending) == 1
    assert pending[0]["status"] == "待补充"


def test_missing_b_and_unknown_filters_do_not_fall_back(live_catalog):
    a_only = live_catalog.query_batch(
        [{
            "query_id": "missing-b",
            "product_terms": ["三通"],
            "specifications": ["50x20"],
            "categories": ["玛钢"],
            "price_markers": ["b"],
        }]
    )["results"][0]
    unknown = live_catalog.query_batch(
        [{
            "query_id": "unknown",
            "product_terms": ["不存在的产品"],
            "specifications": [],
            "categories": [],
            "price_markers": [],
        }]
    )["results"][0]

    assert a_only["status"] == "no_matches"
    assert a_only["candidates"] == []
    assert unknown["status"] == "no_matches"
    assert unknown["unknown_filters"]["product_terms"] == ["不存在的产品"]


def test_explicit_pagination_can_retrieve_every_match_without_silent_loss(live_catalog):
    all_result = live_catalog.query_batch(
        [{"query_id": "all", "product_terms": [], "specifications": [], "categories": [], "price_markers": []}]
    )["results"][0]
    seen = []
    offset = 0
    while True:
        result = live_catalog.query_batch(
            [{
                "query_id": f"page-{offset}",
                "product_terms": [],
                "specifications": [],
                "categories": [],
                "price_markers": [],
                "offset": offset,
                "limit": 5,
            }]
        )["results"][0]
        seen.extend(record["catalog_id"] for record in result["candidates"])
        if not result["has_more"]:
            break
        offset = result["next_offset"]

    assert all_result["returned_count"] == all_result["total_matches"] == 31
    assert seen == [record["catalog_id"] for record in all_result["candidates"]]


@pytest.mark.parametrize(
    "mutator, message",
    [
        (lambda payload: payload["products"][0].__setitem__("unit_price", "1.00"), "安全白名单"),
        (lambda payload: payload["products"][0].__setitem__("catalog_id", "cat_dirty"), "catalog_id 格式"),
        (lambda payload: payload["products"].append(copy.deepcopy(payload["products"][0])), "catalog_id 唯一"),
        (lambda payload: payload["products"][0].__setitem__("price_marker", "B"), "价格标记非法"),
    ],
)
def test_frozen_payload_rejects_private_fields_dirty_ids_duplicates_and_bad_markers(frozen_payload, mutator, message):
    dirty = copy.deepcopy(frozen_payload)
    mutator(dirty)
    with pytest.raises(ValueError, match=message):
        CatalogSnapshot.from_eval_catalog(dirty)


def test_expected_fingerprint_rejects_old_or_modified_snapshot(frozen_payload):
    snapshot = CatalogSnapshot.from_eval_catalog(frozen_payload)
    dirty = copy.deepcopy(frozen_payload)
    dirty["products"][0]["status"] = "启用" if dirty["products"][0]["status"] == "待补充" else "待补充"

    with pytest.raises(ValueError, match="指纹不匹配"):
        CatalogSnapshot.from_eval_catalog(dirty, expected_catalog_fingerprint=snapshot.catalog_fingerprint)


def test_validation_rejects_dirty_visibility_membership_and_field_conflicts(live_catalog):
    result = live_catalog.query_batch(
        [{
            "query_id": "elbow",
            "product_terms": ["弯头"],
            "specifications": ["65"],
            "categories": ["玛钢"],
            "price_markers": ["A"],
        }]
    )["results"][0]
    chosen = result["candidates"][0]
    other = next(record for record in live_catalog.records if record.catalog_id != chosen["catalog_id"])
    items = [
        {
            "item_id": "i1",
            "grade_raw": "A",
            "category": None,
            "product_name": chosen["product_name"],
            "specification": "999",
            "price_marker": "b",
            "candidate_ids": [chosen["catalog_id"], other.catalog_id, "cat_0000000000000000"],
            "selected_candidate_id": chosen["catalog_id"],
            "unresolved": [],
        }
    ]
    report = validate_proposal(items, live_catalog, allowed_catalog_ids={chosen["catalog_id"]})
    codes = {error["code"] for error in report["items"][0]["errors"]}

    assert report["proposal_valid"] is False
    assert {"candidate_id_not_visible", "candidate_id_not_found", "grade_marker_mismatch",
            "selected_category_mismatch", "selected_specification_mismatch",
            "selected_price_marker_mismatch"}.issubset(codes)
    assert report["quote_authorized"] is False


def test_validation_keeps_unknown_grade_as_valid_unresolved_but_never_selectable(live_catalog):
    item = {
        "item_id": "i1",
        "grade_raw": "陌生袋色",
        "category": None,
        "product_name": None,
        "specification": "25",
        "price_marker": None,
        "candidate_ids": [],
        "selected_candidate_id": None,
        "unresolved": ["等级未知"],
    }
    report = validate_proposal([item], live_catalog, allowed_catalog_ids=set())

    assert report["proposal_valid"] is True
    assert report["items"][0]["errors"] == []
    assert {issue["code"] for issue in report["items"][0]["unresolved"]} == {
        "model_declared_unresolved", "unknown_grade"
    }
    assert report["quote_authorized"] is False


def test_validation_pending_selection_stays_unresolved_and_alias_name_is_accepted(live_catalog):
    result = live_catalog.query_batch(
        [{
            "query_id": "pending",
            "product_terms": ["弯头"],
            "specifications": ["11422.5"],
            "categories": ["沟槽"],
            "price_markers": ["b"],
        }]
    )["results"][0]
    record = result["candidates"][0]
    item = {
        "item_id": "i1",
        "grade_raw": None,
        "category": record["category"],
        "product_name": "弯头",
        "specification": record["specification"],
        "price_marker": record["price_marker"],
        "candidate_ids": [record["catalog_id"]],
        "selected_candidate_id": record["catalog_id"],
        "unresolved": ["主数据待补充"],
    }
    report = validate_proposal([item], live_catalog, allowed_catalog_ids={record["catalog_id"]})

    assert report["proposal_valid"] is True
    assert report["items"][0]["errors"] == []
    assert "selected_record_pending" in {issue["code"] for issue in report["items"][0]["unresolved"]}
    assert report["quote_authorized"] is False
