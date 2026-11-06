"""Offline regression coverage for source-scoped catalog and grade defaults."""
import json

import pytest

from src.agent_catalog import CatalogRecord, CatalogSnapshot, inquiry_categories
from src.agent_context import inquiry_contexts, item_context
from src.agent_inquiry import _records_from_report
from src.agent_workflow import make_payload, run_inquiry
from src.agent_inquiry import RUN_CONFIG
from src.core import confirm_inquiry, normalize_price_marker


GRADE_MAPPINGS = {"中": "a", "黑袋": "a", "蓝袋": "b", "好": "b"}


def _record(ident, category, product, marker="b"):
    return CatalogRecord(ident, category, product, "25", marker, (), "启用")


def _snapshot(records, mappings=GRADE_MAPPINGS):
    return CatalogSnapshot(tuple(records), tuple(mappings.items()), "synthetic")


def _document(texts, *, name="synthetic.txt"):
    units = []
    for index, value in enumerate(texts, 1):
        if isinstance(value, dict):
            units.append({"unit_id": f"u{index}", **value})
        else:
            units.append({"unit_id": f"u{index}", "text": value, "page": 1})
    return {"source_id": "source-1", "source_type": "text", "source_name": name, "units": units}


def _convert(*, texts, raw_grade=None, marker=None, category="玛钢", product="弯头",
             mappings=GRADE_MAPPINGS, evidence=None, selected=True, status=None, unresolved=(),
             document_unresolved=(), coverage_reason=None):
    """Run deterministic catalog validation and the real UI-record conversion."""
    document = _document(texts)
    expected_marker = normalize_price_marker(raw_grade, dict(mappings)) if raw_grade else marker
    catalog_record = _record("cat_0000000000000001", category or "玛钢", product,
                             expected_marker if expected_marker in {"a", "A", "b"} else "b")
    catalog = _snapshot([catalog_record], mappings)
    if evidence is None:
        target = document["units"][-1]
        evidence = [{"unit_id": target["unit_id"], "quote": target["text"]}]
    candidate_ids = [catalog_record.catalog_id] if selected else []
    item = {
        "item_id": "i1", "source_id": document["source_id"], "evidence": evidence,
        "product_name_raw": product, "specification_raw": "25", "quantity_raw": "10件",
        "grade_raw": raw_grade, "category": category, "product_name": product,
        "specification": "25", "quantity_cartons": 10, "price_marker": marker,
        "candidate_ids": candidate_ids,
        "selected_candidate_id": catalog_record.catalog_id if selected else None,
        "unresolved": list(unresolved),
    }
    catalog_validation = catalog.validate_proposal([item], allowed_catalog_ids=set(candidate_ids))
    item_check = catalog_validation["items"][0]
    coverage = [] if coverage_reason is None else [{
        "unit_id": document["units"][-1]["unit_id"], "disposition": "unresolved",
        "item_ids": [item["item_id"]], "reason": coverage_reason,
    }]
    validation = {
        "parsed": {"document_type": "inquiry", "items": [item],
                   "unresolved": list(document_unresolved), "coverage": coverage},
        "catalog_validation": catalog_validation, "structure_errors": [], "business_errors": catalog_validation["errors"],
        "unresolved": [*document_unresolved, *(issue["message"] for issue in item_check["unresolved"]),
                       *([] if coverage_reason is None else [coverage_reason])],
    }
    sample_status = status or ("unresolved" if validation["unresolved"] else "proposal_valid")
    report = {"samples": [{"status": sample_status, "validation": validation}]}
    records, _ = _records_from_report([document], report, dict(mappings))
    return records[0], item_check, catalog_validation


@pytest.mark.parametrize("scope", ["document", "item", "coverage"])
@pytest.mark.parametrize(("raw", "expected"), [("黑", "a"), ("蓝", "b"), ("A", "A"), (None, "b")])
def test_unrelated_warning_words_do_not_clear_valid_or_default_grade(scope, raw, expected):
    message = "重复识别说明引用黑/蓝原字；品名尚未确定，目录档位含 a/A/b，袋色等级不是本条未决原因"
    options = {"document_unresolved": [message]} if scope == "document" else (
        {"unresolved": [message]} if scope == "item" else {"coverage_reason": message})
    text = "25弯头10件" + (raw or "")
    record, check, _ = _convert(texts=[text], raw_grade=raw, **options)
    assert check["resolved_price_marker"] == record.price_marker == expected
    assert record.original_grade_text == raw and not record.grade_review_required
    assert message in record.warnings


@pytest.mark.parametrize("prefix", ["等级：", "  等级:"])
@pytest.mark.parametrize("scope", ["item", "coverage"])
def test_explicit_item_grade_uncertainty_still_requires_manual_selection(prefix, scope):
    message = prefix + "此项黑蓝标注冲突，需人工选择"
    options = {"unresolved": [message]} if scope == "item" else {"coverage_reason": message}
    record, _, _ = _convert(texts=["25弯头10件黑"], raw_grade="黑", **options)
    assert record.original_grade_text == "黑"
    assert record.price_marker is None and record.grade_review_required
    with pytest.raises(ValueError):
        confirm_inquiry(record)


def test_document_grade_label_does_not_substitute_for_item_grade_evidence():
    message = "等级：文档汇总引用黑蓝文字，重复行归属未决"
    record, _, _ = _convert(texts=["25弯头10件蓝"], raw_grade="蓝", document_unresolved=[message])
    assert record.price_marker == "b" and not record.grade_review_required
    assert message in record.warnings


def test_unclassified_source_starts_with_magang_index_and_wide_pagination_stays_scoped():
    catalog = _snapshot([
        _record("cat_0000000000000001", "玛钢", "弯头"),
        _record("cat_0000000000000002", "玛钢", "三通"),
        _record("cat_0000000000000003", "沟槽", "卡箍"),
        _record("cat_0000000000000004", "球铁", "球铁三通"),
    ])
    plain_source = _document(["25弯头10件"], name="plain.txt")
    mixed_source = _document(["沟槽", "球热", "QT 三通"], name="mixed.txt")
    plain_categories = inquiry_categories(plain_source)
    mixed_categories = inquiry_categories(mixed_source)
    assert plain_categories == {"玛钢"}
    assert mixed_categories == {"玛钢", "沟槽", "球铁"}

    scoped = catalog.restrict_categories(plain_categories)
    payload = make_payload(
        {"document": plain_source, "catalog": {"grade_mappings": []}}, "offline", "B", scoped
    )
    initial = json.loads(payload["messages"][1]["content"])["catalog"]["index"]
    assert initial["record_count"] == 2
    assert {row["category"] for row in initial["products"]} == {"玛钢"}

    first = scoped.query_batch([{"query_id": "wide-1", "product_terms": [], "specifications": [],
                                 "categories": [], "price_markers": [], "offset": 0, "limit": 1}])["results"][0]
    second = scoped.query_batch([{"query_id": "wide-2", "product_terms": [], "specifications": [],
                                  "categories": [], "price_markers": [], "offset": 1, "limit": 1}])["results"][0]
    assert first["has_more"] and first["candidates"][0]["category"] == "玛钢"
    assert not second["has_more"] and second["candidates"][0]["category"] == "玛钢"
    assert {row.category for row in catalog.restrict_categories(mixed_categories).records} == {
        "玛钢", "沟槽", "球铁",
    }


def test_actual_workflow_uses_each_source_scope_for_initial_and_tool_turns(tmp_path, monkeypatch):
    catalog = _snapshot([_record("cat_0000000000000001", "玛钢", "弯头"),
                         _record("cat_0000000000000002", "沟槽", "卡箍"),
                         _record("cat_0000000000000003", "球铁", "球铁三通")])
    monkeypatch.setattr(CatalogSnapshot, "assert_current", lambda self: "synthetic-master")
    documents, seen = [], []
    expected = {"source-1": {"玛钢"}, "source-2": {"玛钢", "沟槽"}, "source-3": {"玛钢", "球铁"}}
    for number, text in enumerate(("25弯头10件", "沟槽 卡箍25 1箱", "QT 球热 三通25 1箱"), 1):
        documents.append({**_document([text]), "source_id": f"source-{number}", "images": []})

    def transport(payload, key, output, **limits):
        supplied = json.loads(payload["messages"][1]["content"])
        document = supplied["document"]
        ident = document["source_id"]
        assert {row["category"] for row in supplied["catalog"]["index"]["products"]} == expected[ident]
        tool_messages = [message for message in payload["messages"] if message["role"] == "tool"]
        message = {"role": "assistant", "content": "", "reasoning_content": ""}
        if not tool_messages:
            query = {"queries": [{"query_id": "all", "product_terms": [], "specifications": [],
                                   "categories": [], "price_markers": []}]}
            message["tool_calls"] = [{"id": "query", "type": "function", "function": {
                "name": "query_batch", "arguments": json.dumps(query)}}]
        else:
            result = json.loads(tool_messages[-1]["content"])["results"][0]
            assert {row["category"] for row in result["candidates"]} == expected[ident]
            seen.append(ident)
            message["content"] = json.dumps({"document_type": "unknown", "items": [], "unresolved": [],
                "coverage": [{"source_id": ident, "unit_id": unit["unit_id"], "disposition": "context",
                              "item_ids": [], "reason": "合成目录范围检查"} for unit in document["units"]]})
        return {"model": "offline", "usage": {}, "finish_reason": "stop" if tool_messages else "tool_calls", "message": message}

    report = run_inquiry(documents, "offline", tmp_path / "run", config=dict(RUN_CONFIG),
                         master_snapshot=catalog, transport=transport, environ={"DEEPSEEK_API_KEY": "offline-only"})
    assert seen == list(expected) and report["model_requests"] == 6
    assert all(sample["status"] == "proposal_valid" for sample in report["samples"])


def test_empty_source_restriction_does_not_fall_back_to_full_catalog():
    catalog = _snapshot([
        _record("cat_0000000000000001", "沟槽", "卡箍"),
        _record("cat_0000000000000002", "球铁", "球铁三通"),
    ])
    source = _document(["25弯头10件"])
    restricted = catalog.restrict_categories(inquiry_categories(source))
    assert restricted.records == ()
    result = restricted.query_batch([{"query_id": "empty-wide", "product_terms": [],
                                      "specifications": [], "categories": [], "price_markers": []}])["results"][0]
    assert result["status"] == "no_matches" and result["candidates"] == []


def test_missing_category_and_grade_default_independently_of_unknown_product():
    record, _, _ = _convert(texts=["陌生货25 10件"], category=None, product="陌生货",
                            selected=False, unresolved=["产品名待核对"])
    assert (record.category, record.price_marker, record.original_grade_text) == ("玛钢", "b", None)
    assert not record.grade_review_required and not record.manually_confirmed


@pytest.mark.parametrize(
    ("raw", "proposed", "expected"),
    [(None, None, "b"), (None, "b", "b"),
     ("A", "A", "A"), ("a", "a", "a"), ("b", "b", "b"),
     ("黑", None, "a"), ("蓝", None, "b")],
)
def test_resolved_grade_reaches_confirmation_row_without_fabricating_raw_text(raw, proposed, expected):
    record, check, _ = _convert(texts=["25弯头10件"], raw_grade=raw, marker=proposed)
    assert check["resolved_price_marker"] == expected
    assert record.price_marker == expected and record.original_grade_text == raw
    assert not record.grade_review_required
    if raw is None and proposed is None:
        confirm_inquiry(record)
        assert record.price_marker == "b"


@pytest.mark.parametrize("claimed", ["A", "a"])
def test_rawless_a_or_uppercase_a_claim_stays_gated(claimed):
    record, _, _ = _convert(texts=["25弯头10件"], raw_grade=None, marker=claimed)
    assert record.original_grade_text is None
    assert record.price_marker is None and record.grade_review_required
    with pytest.raises(ValueError):
        confirm_inquiry(record)


def test_unknown_conflict_and_failure_placeholder_keep_confirmation_gates():
    unknown, unknown_check, _ = _convert(
        texts=["25弯头10件"], raw_grade="紫袋", marker=None, mappings=GRADE_MAPPINGS,
        selected=False,
    )
    assert unknown_check["resolved_price_marker"] is None
    assert unknown.original_grade_text == "紫袋" and unknown.price_marker is None
    with pytest.raises(ValueError):
        confirm_inquiry(unknown)

    conflict, _, _ = _convert(texts=["L中=25弯头10件"], raw_grade="蓝", marker=None)
    assert conflict.original_grade_text == "蓝"
    assert conflict.price_marker is None and conflict.grade_review_required
    with pytest.raises(ValueError):
        confirm_inquiry(conflict)

    failed, _ = _records_from_report(
        [_document(["25弯头10件"])],
        {"samples": [{"status": "transport_failed", "validation": None}]}, GRADE_MAPPINGS,
    )
    assert failed[0].extraction_method == "待人工处理"
    assert failed[0].price_marker is None and failed[0].product_name is None
    with pytest.raises(ValueError):
        confirm_inquiry(failed[0])


def test_category_ambiguity_clears_category_without_suppressing_true_no_grade_default():
    texts = ["沟槽 25弯头10件", "球铁 25弯头10件"]
    evidence = [{"unit_id": "u1", "quote": texts[0]}, {"unit_id": "u2", "quote": texts[1]}]
    record, _, _ = _convert(texts=texts, evidence=evidence)
    assert record.category is None
    assert record.price_marker == "b" and record.original_grade_text is None


def test_title_grade_is_scoped_to_same_page_column_and_survives_category_heading():
    document = _document([
        {"text": "L中=", "page": 1, "reading_method": "local_ocr", "block_label": "paragraph_title",
         "bbox": [1000, 0, 1200, 40]},
        {"text": "25弯头10件", "page": 1, "reading_method": "local_ocr", "bbox": [1010, 100, 1210, 160]},
        {"text": "25弯头5件", "page": 1, "reading_method": "local_ocr", "bbox": [0, 100, 200, 160]},
        {"text": "32三通5件", "page": 2, "reading_method": "local_ocr", "bbox": [1010, 100, 1210, 160]},
    ])
    contexts = inquiry_contexts(document)

    def context(unit_id, quote, product="弯头"):
        return item_context({"product_name_raw": product,
                             "evidence": [{"unit_id": unit_id, "quote": quote}]}, contexts)

    assert context("u2", "25弯头10件")[1:] == ("中", False, False)
    assert context("u3", "25弯头5件")[1:] == (None, False, False)
    assert context("u4", "32三通5件", "三通")[1:] == (None, False, False)
    assert normalize_price_marker("中", GRADE_MAPPINGS) == "a"

    plain = _document(["L中=报价档位", "球铁", "32三通5件"])
    plain_contexts = inquiry_contexts(plain)
    category, grade, category_ambiguous, grade_ambiguous = item_context(
        {"product_name_raw": "三通", "evidence": [{"unit_id": "u3", "quote": "32三通5件"}]},
        plain_contexts,
    )
    assert (category, grade, category_ambiguous, grade_ambiguous) == ("球铁", "中", False, False)

    inherited, _, _ = _convert(
        texts=["L中=报价档位", "球铁", "32三通5件"], raw_grade=None, marker="a",
        category="球铁", product="三通",
    )
    assert inherited.category == "球铁" and inherited.price_marker == "a"
    assert inherited.original_grade_text is None and not inherited.grade_review_required
