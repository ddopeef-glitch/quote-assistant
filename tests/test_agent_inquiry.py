"""Exercise the actual UI route with offline transports; never call a provider."""
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace

import fitz
from openpyxl import Workbook
from PIL import Image
import pytest
from streamlit.testing.v1 import AppTest

from src import agent_inquiry as inquiry
from src import agent_workflow as workflow
from src.core import ProductRecord, confirm_inquiry, discover_master_data
from src.input_router import extract_evidence_units


TEXT = "沟槽 90°弯头 108 b 1箱"
MASTER = discover_master_data(inquiry.ROOT / "主数据")


def response(content="", calls=None):
    message = {"role": "assistant", "content": content, "reasoning_content": "离线测试推理"}
    if calls:
        message["tool_calls"] = calls
    return {"model": "deepseek-v4-pro", "usage": {"prompt_tokens": 20, "completion_tokens": 10},
            "finish_reason": "tool_calls" if calls else "stop", "message": message}


def proposal(document, candidate):
    ident = candidate["catalog_id"]
    item = {"item_id": "i1", "source_id": document["source_id"],
            "evidence": [{"unit_id": "u1", "quote": document["units"][0]["text"]}],
            "product_name_raw": "90°弯头", "specification_raw": "108", "quantity_raw": "1箱", "grade_raw": "b",
            "category": candidate["category"], "product_name": candidate["product_name"],
            "specification": candidate["specification"], "quantity_cartons": 1, "price_marker": "b",
            "candidate_ids": [ident], "selected_candidate_id": ident, "unresolved": []}
    return {"document_type": "inquiry", "items": [item], "unresolved": [],
            "coverage": [{"source_id": document["source_id"], "unit_id": unit["unit_id"],
                          "disposition": "items" if index == 0 else "context",
                          "item_ids": ["i1"] if index == 0 else [], "reason": "产品行" if index == 0 else "标题"}
                         for index, unit in enumerate(document["units"])]}


def model_transport(payloads, change=None):
    def transport(payload, key, output, **limits):
        payloads.append(payload)
        assert (limits["timeout_seconds"] is None or 0 < limits["timeout_seconds"] <= 180) and limits["idle_seconds"] == 60
        document = json.loads(payload["messages"][1]["content"])["document"]
        assert document["images"] == []
        tools = [message for message in payload["messages"] if message["role"] == "tool"]
        if not tools:
            query = {"queries": [{"query_id": "q1", "product_terms": ["90°弯头"], "specifications": ["108"],
                                  "categories": ["沟槽"], "price_markers": ["b"]}]}
            return response(calls=[{"id": "query-1", "type": "function", "function": {
                "name": "query_batch", "arguments": workflow.dumps(query)}}])
        candidate = json.loads(tools[-1]["content"])["results"][0]["candidates"][0]
        answer = proposal(document, candidate)
        if change:
            change(answer)
        return response(workflow.dumps(answer))
    return transport


def run(tmp_path, transport, *, files=(), text=TEXT, **kwargs):
    return inquiry.recognize_inquiry(files, text, MASTER, None, transport=transport,
                                    environ={"DEEPSEEK_API_KEY": "offline-secret"}, output_root=tmp_path, **kwargs)


def test_real_catalog_loop_produces_unconfirmed_records_and_audit(tmp_path):
    payloads = []
    records, audit = run(tmp_path, model_transport(payloads))
    assert len(payloads) == 2 and audit["run"]["samples"][0]["status"] == "proposal_valid"
    record = records[0]
    assert (record.category, record.product_name, record.specification, record.quantity_cartons) == ("沟槽", "90°弯头", "108", 1)
    assert record.original_grade_text == record.price_marker == "b"
    assert not record.manually_confirmed and record.confirmed_inquiry is None
    assert record.unit_price is record.packing_quantity is record.master_packing_quantity is None
    assert record.original_text == TEXT
    serialized = json.dumps(payloads, ensure_ascii=False)
    assert all(field not in serialized for field in ("unit_price", "packing_quantity", "blank_weight", "offline-secret"))
    assert audit["run"]["samples"][0]["validation"]["parsed"]["items"][0]["selected_candidate_id"]
    assert Path(audit["run"]["evidence_directory"], "source-1", "round-02", "validation.json").is_file()
    confirm_inquiry(record)
    assert record.manually_confirmed


def test_unknown_grade_stays_unresolved_and_blocks_confirmation(tmp_path):
    def change(answer):
        answer["items"][0].update(grade_raw="陌生档", price_marker=None, selected_candidate_id=None,
                                  unresolved=["等级需人工核对"])
    records, audit = run(tmp_path, model_transport([], change))
    assert audit["run"]["samples"][0]["status"] == "unresolved"
    assert records[0].original_grade_text == "陌生档" and records[0].price_marker is None
    with pytest.raises(ValueError):
        confirm_inquiry(records[0])


@pytest.mark.parametrize("failure", ["coverage", "candidate", "json"])
def test_invalid_proposal_blocks_fields_and_remaining_sources(tmp_path, failure):
    calls = []
    def change(answer):
        if failure == "coverage":
            answer["coverage"] = []
        else:
            answer["items"][0]["product_name"] = "目录不存在的货品"
    transport = model_transport(calls, change)
    if failure == "json":
        def transport(*args, **kwargs):
            calls.append(1)
            return response("not JSON")
    records, audit = run(tmp_path, transport, files=[("first.txt", TEXT.encode("utf-8"))])
    assert audit["run"]["samples"][1]["status"] == "not_processed"
    assert len(records) == 2 and all(record.product_name is None and record.quantity_cartons is None for record in records)
    assert len(audit["notices"]) == 2
    for record in records:
        with pytest.raises(ValueError):
            confirm_inquiry(record)


def test_unresolved_unit_without_item_is_preserved_but_context_is_not_product(tmp_path):
    def transport(payload, *args, **kwargs):
        document = json.loads(payload["messages"][1]["content"])["document"]
        return response(workflow.dumps({"document_type": "inquiry", "items": [], "unresolved": [],
            "coverage": [{"source_id": document["source_id"], "unit_id": unit["unit_id"], "item_ids": [],
                          "disposition": "unresolved" if unit["unit_id"] == "u1" else "context", "reason": "未知货品待核对"}
                         for unit in document["units"]]}))
    records, audit = run(tmp_path, transport, text="不在旧规则中的陌生货\n标题上下文")
    assert len(records) == 1 and records[0].original_text == "不在旧规则中的陌生货"
    assert records[0].product_name is None
    assert len(audit["documents"][0]["units"]) == 2


@pytest.mark.parametrize("text, source_count", [(TEXT + "\n标题上下文", 3), ("  \n  ", 2)])
def test_each_upload_and_nonempty_paste_has_two_independent_requests(tmp_path, text, source_count):
    calls = []
    records, audit = run(tmp_path, model_transport(calls), text=text,
                         files=[("same.txt", TEXT.encode()), ("same.txt", TEXT.encode())])
    assert inquiry.RUN_CONFIG["max_requests"] is None and inquiry.RUN_CONFIG["max_calls_per_sample"] == 2
    assert len(calls) == audit["run"]["model_requests"] == source_count * 2
    assert [sample["status"] for sample in audit["run"]["samples"]] == ["proposal_valid"] * source_count
    assert [sample["model_requests"] for sample in audit["run"]["samples"]] == [2] * source_count
    assert [document["source_id"] for document in audit["documents"]] == [f"source-{i}" for i in range(1, source_count + 1)]
    assert len(records) == source_count and all(record.product_name == "90°弯头" for record in records)
    assert not audit["notices"] and all("本来源最多可用 2 次" in payload["messages"][0]["content"] for payload in calls)
    ledger = [json.loads(line) for line in Path(audit["run"]["evidence_directory"], "ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    reserved = [row for row in ledger if row["event"] == "reserve"]
    assert [row["ordinal"] for row in reserved] == list(range(1, source_count * 2 + 1))
    assert [row["round"] for row in reserved] == [1, 2] * source_count
    assert all(not row["network_retry"] for sample in audit["run"]["samples"] for row in sample["rounds"])


@pytest.mark.parametrize("outcome", ["unresolved", "request_failed"])
def test_network_retry_stage_and_source_request_limit(tmp_path, outcome):
    calls, stages = [], []
    def transport(payload, key, output, **limits):
        calls.append(output)
        assert limits == {"timeout_seconds": None, "idle_seconds": 60}
        if len(calls) == 1 or outcome == "request_failed":
            raise workflow.TransportError("incomplete_http_read", retryable=True)
        document = json.loads(payload["messages"][1]["content"])["document"]
        return response(workflow.dumps({"document_type": "inquiry", "items": [], "unresolved": [],
            "coverage": [{"source_id": document["source_id"], "unit_id": unit["unit_id"],
                          "disposition": "unresolved", "item_ids": [], "reason": "需要人工核对"}
                         for unit in document["units"]]}))
    records, audit = run(tmp_path, transport, on_stage=stages.append)
    sample = audit["run"]["samples"][0]
    assert len(calls) == audit["run"]["model_requests"] == 2
    assert sample["status"] == outcome
    assert [row["network_retry"] for row in sample["rounds"]] == [False, True]
    assert stages == ["正在读取询价文件…", "模型正在推理…", "模型网络请求失败，正在重试（本来源请求 2/2）…"]
    assert not Path(audit["run"]["evidence_directory"], "source-1", "round-03").exists()
    assert all(record.product_name is None and not record.manually_confirmed for record in records)
    if outcome == "request_failed":
        assert sample["retry_stop_reason"] == "source_request_limit"
        assert "响应未完整接收" in audit["notices"][0] and "次数已用尽，未再重试" in audit["notices"][0]
    else:
        assert "retry_stop_reason" not in sample


@pytest.mark.parametrize("first_request", ["tool", "previous_source"])
def test_explicit_batch_limit_still_blocks_third_request(tmp_path, first_request):
    calls, stages = [], []
    delegate = model_transport([])
    def transport(payload, key, output, **limits):
        calls.append(output)
        if len(calls) == 2:
            raise workflow.TransportError("truncated_sse", retryable=True)
        if first_request == "tool":
            return delegate(payload, key, output, **limits)
        document = json.loads(payload["messages"][1]["content"])["document"]
        return response(workflow.dumps({"document_type": "inquiry", "items": [], "unresolved": [],
            "coverage": [{"source_id": document["source_id"], "unit_id": unit["unit_id"],
                          "disposition": "unresolved", "item_ids": [], "reason": "需要人工核对"}
                         for unit in document["units"]]}))
    records, audit = run(tmp_path, transport, files=[("one.txt", TEXT.encode()), ("two.txt", TEXT.encode())],
                         config={**inquiry.RUN_CONFIG, "max_requests": 2}, on_stage=stages.append)
    samples = audit["run"]["samples"]
    failed = samples[0 if first_request == "tool" else 1]
    assert len(calls) == audit["run"]["model_requests"] == 2
    assert failed["status"] == "request_failed" and failed["retry_stop_reason"] == "batch_request_limit"
    assert samples[-1]["status"] == "not_processed"
    assert all(not row["network_retry"] for sample in samples for row in sample.get("rounds", []))
    assert not any("正在重试" in stage for stage in stages)
    assert any("响应未完整接收" in notice and "次数已用尽，未再重试" in notice for notice in audit["notices"])
    assert len(records) == 3 and all(record.product_name is None for record in records)
    for record in records:
        with pytest.raises(ValueError):
            confirm_inquiry(record)


@pytest.mark.parametrize("first_requests", [1, 2])
@pytest.mark.parametrize("failure", ["tools", "network"])
def test_source_exhaustion_keeps_failed_evidence_and_next_source_allowance(tmp_path, first_requests, failure):
    calls, stages = [], []
    delegate = model_transport([])
    def transport(payload, key, output, **limits):
        document = json.loads(payload["messages"][1]["content"])["document"]
        ident = document["source_id"]
        calls.append(ident)
        if ident == "source-1" and first_requests == 1:
            return response(workflow.dumps({"document_type": "unknown", "items": [], "unresolved": [],
                "coverage": [{"source_id": ident, "unit_id": "u1", "disposition": "non_inquiry",
                              "item_ids": [], "reason": "仅上下文"}]}))
        if ident == "source-2":
            if failure == "network":
                raise workflow.TransportError("incomplete_http_read", retryable=True)
            number = calls.count(ident)
            query = {"queries": [{"query_id": f"q{number}", "product_terms": ["90°弯头"],
                                  "specifications": ["108" if number == 1 else "114"],
                                  "categories": ["沟槽"], "price_markers": ["b"]}]}
            return response(calls=[{"id": f"query-{number}", "type": "function", "function": {
                "name": "query_batch", "arguments": workflow.dumps(query)}}])
        return delegate(payload, key, output, **limits)
    records, audit = run(tmp_path, transport, files=[("one.txt", TEXT.encode()), ("two.txt", TEXT.encode())],
                         on_stage=stages.append)
    samples = audit["run"]["samples"]
    assert calls == ["source-1"] * first_requests + ["source-2"] * 2 + ["source-3"] * 2
    assert audit["run"]["model_requests"] == first_requests + 4
    assert [sample["model_requests"] for sample in samples] == [first_requests, 2, 2]
    assert [sample["status"] for sample in samples] == ["proposal_valid", "call_limit" if failure == "tools" else "request_failed", "proposal_valid"]
    pending = next(record for record in records if record.source_name == "two.txt")
    assert pending.original_text == TEXT and pending.extraction_method == "待人工处理"
    assert pending.product_name is None and any("本来源的模型请求次数已用尽" in notice for notice in audit["notices"])
    with pytest.raises(ValueError):
        confirm_inquiry(pending)
    assert records[-1].source_name == "粘贴文字.txt" and records[-1].product_name == "90°弯头"
    retries = [stage for stage in stages if "正在重试" in stage]
    assert retries == (["模型网络请求失败，正在重试（本来源请求 2/2）…"] if failure == "network" else [])
    assert [row["network_retry"] for row in samples[1]["rounds"]] == [False, failure == "network"]
    if failure == "network":
        assert samples[1]["retry_stop_reason"] == "source_request_limit"


@pytest.mark.parametrize("failure", ["HTTP_401", "HTTP_402", "HTTP_403"])
def test_credential_balance_and_permission_failures_still_stop_all_sources(tmp_path, failure):
    calls = []
    def transport(*args, **kwargs):
        calls.append(1)
        raise workflow.TransportError(failure)
    records, audit = run(tmp_path, transport, files=[("first.txt", TEXT.encode())])
    assert len(calls) == 1
    assert [sample["status"] for sample in audit["run"]["samples"]] == ["request_failed", "not_processed"]
    assert all(record.product_name is None and record.extraction_method == "待人工处理" for record in records)


@pytest.mark.parametrize("file_kind", ["xlsx", "pdf"])
def test_sheets_and_pages_do_not_create_request_allowances(tmp_path, file_kind):
    if file_kind == "xlsx":
        book = Workbook()
        book.active["A1"] = "first row"
        book.active["A2"] = "second row"
        book.create_sheet("Second")["A1"] = "third row"
        buffer = BytesIO()
        book.save(buffer)
        data = buffer.getvalue()
    else:
        with fitz.open() as pdf:
            for index in range(3):
                pdf.new_page().insert_text((30, 40), f"unknown product {index}")
            data = pdf.tobytes()
    calls = []
    def transport(payload, *args, **kwargs):
        document = json.loads(payload["messages"][1]["content"])["document"]
        calls.append(document)
        raise workflow.TransportError("truncated_sse", retryable=True)
    records, audit = run(tmp_path, transport, files=[(f"one.{file_kind}", data)], text="")
    assert len(calls) == audit["run"]["model_requests"] == 2
    assert len(audit["run"]["samples"]) == 1 and len(audit["documents"][0]["units"]) == 3
    assert all(document["source_id"] == "source-1" and len(document["units"]) == 3 for document in calls)
    assert len(records) == 1 and all(unit["text"] in records[0].original_text for unit in calls[0]["units"])


@pytest.mark.parametrize("gate, expected", [("budget", "budget_limit"), ("total_seconds", "batch_limit"),
                                           ("sample_seconds", "sample_timeout"), ("master", "stale_source")])
def test_retry_stage_is_not_shown_when_preflight_stops_it(tmp_path, monkeypatch, gate, expected):
    calls, stages, clock = [], [], [1000.0]
    monkeypatch.setattr(workflow.time, "time", lambda: clock[0])
    monkeypatch.setattr(workflow.time, "monotonic", lambda: clock[0])
    config = {**inquiry.RUN_CONFIG}
    if gate == "budget":
        # Fix this synthetic reservation threshold independently of live rates.
        config.update(budget="3", input_per_million="9", output_per_million="27")
    elif gate != "master":
        config[gate] = 5
    original = inquiry.CatalogSnapshot.assert_current
    def assert_current(snapshot):
        original(snapshot)
        if gate == "master" and calls:
            raise ValueError("synthetic snapshot change")
    monkeypatch.setattr(inquiry.CatalogSnapshot, "assert_current", assert_current)
    def transport(*args, **kwargs):
        calls.append(1)
        clock[0] += 10
        raise workflow.TransportError("truncated_sse", retryable=True)
    if gate == "master":
        with pytest.raises(inquiry.InquiryError, match="未能完成"):
            run(tmp_path, transport, config=config, on_stage=stages.append)
    else:
        _, audit = run(tmp_path, transport, config=config, on_stage=stages.append)
        assert "未再重试" in audit["notices"][0]
    assert calls == [1] and not any("正在重试" in stage for stage in stages)
    sample = json.loads(next(tmp_path.glob("*/source-1/summary.json")).read_text(encoding="utf-8"))
    assert sample["status"] == sample["retry_stop_reason"] == expected
    assert sample["rounds"][0]["network_retry"] is False


def test_reservation_limit_prevents_any_paid_request(tmp_path):
    config = {**inquiry.RUN_CONFIG, "budget": "0.01"}
    records, audit = run(tmp_path, lambda *a, **k: pytest.fail("budget must block call"), config=config)
    assert audit["run"]["model_requests"] == 0
    assert audit["run"]["samples"][0]["status"] == "budget_limit"
    assert records[0].quantity_cartons is None


def test_shared_deadline_stops_continuation_and_remaining_sources(tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(workflow.time, "time", lambda: clock[0])
    calls = []
    delegate = model_transport(calls)
    def transport(*args, **kwargs):
        result = delegate(*args, **kwargs)
        clock[0] += 601
        return result
    records, audit = run(tmp_path, transport, files=[("one.txt", TEXT.encode())],
                         config={**inquiry.RUN_CONFIG, "total_seconds": 600, "sample_seconds": 600, "request_seconds": 180})
    assert len(calls) == 1 and audit["run"]["samples"][1]["status"] == "not_processed"
    assert len(records) == 2 and all(record.product_name is None for record in records)


def test_slow_ocr_and_model_continuation_have_no_ui_cumulative_deadline(tmp_path, monkeypatch):
    clock, stages, calls = [1000.0], [], []
    monkeypatch.setattr(workflow.time, "time", lambda: clock[0])
    monkeypatch.setattr(workflow.time, "monotonic", lambda: clock[0])
    image = BytesIO()
    Image.new("RGB", (24, 24), "white").save(image, "PNG")

    def extract_raw(data):
        assert stages[-1] == "正在识别图片…"
        clock[0] += 901
        return [{"res": {"parsing_res_list": [{"block_content": TEXT}]}}]

    delegate = model_transport(calls)
    def transport(*args, **kwargs):
        assert stages[-1] == "模型正在推理…"
        assert kwargs["timeout_seconds"] is None and kwargs["idle_seconds"] == 60
        result = delegate(*args, **kwargs)
        clock[0] += 601
        return result

    records, audit = inquiry.recognize_inquiry(
        [("slow.png", image.getvalue())], "", MASTER, SimpleNamespace(extract_raw=extract_raw),
        transport=transport, environ={"DEEPSEEK_API_KEY": "offline-secret"}, output_root=tmp_path, on_stage=stages.append,
    )
    assert len(calls) == 2 and audit["run"]["samples"][0]["status"] == "proposal_valid"
    assert records[0].product_name == "90°弯头" and not records[0].manually_confirmed
    assert stages == ["正在读取询价文件…", "正在识别图片…", "模型正在推理…", "正在核对产品目录…", "模型正在推理…"]
    saved = json.loads(Path(audit["run"]["evidence_directory"], "run.json").read_text(encoding="utf-8"))
    assert saved["config"]["total_seconds"] is None and saved["config"]["sample_seconds"] is None


def test_final_snapshot_change_discards_all_proposals(tmp_path, monkeypatch):
    calls = []
    original = inquiry.CatalogSnapshot.assert_current
    checks = []
    def assert_current(snapshot):
        original(snapshot)
        checks.append(1)
        if len(checks) == 6:
            raise ValueError("master changed after final response")
    monkeypatch.setattr(inquiry.CatalogSnapshot, "assert_current", assert_current)
    with pytest.raises(inquiry.InquiryError, match="全部作废"):
        run(tmp_path, model_transport(calls))
    saved = next(tmp_path.glob("*/summary.json"))
    assert json.loads(saved.read_text(encoding="utf-8"))["samples"][0]["status"] == "stale_source"


def test_no_candidate_item_is_preserved_and_price_matrix_is_not_quantity(tmp_path):
    def no_candidate(answer):
        answer["items"][0].update(product_name_raw="陌生货", product_name="陌生货", candidate_ids=[], selected_candidate_id=None,
                                  unresolved=["目录无候选，保持原文"])
    records, audit = run(tmp_path, model_transport([], no_candidate))
    assert records[0].product_name == "陌生货" and records[0].quantity_cartons == 1
    assert audit["run"]["samples"][0]["status"] == "unresolved"
    records, audit = run(tmp_path, model_transport([], lambda answer: answer.update(document_type="price_matrix")))
    assert records[0].quantity_cartons is None and "价格矩阵" in audit["notices"][0]


def test_no_credentials_or_input_failure_never_calls_model(tmp_path, monkeypatch):
    with pytest.raises(inquiry.InquiryError, match="DEEPSEEK_API_KEY"):
        inquiry.recognize_inquiry([], TEXT, MASTER, None, environ={}, transport=lambda *a, **k: pytest.fail())
    monkeypatch.setattr(inquiry, "extract_evidence_units", lambda *a, **k: (_ for _ in ()).throw(ImportError("sensitive error")))
    with pytest.raises(inquiry.InquiryError, match="另存为 .xlsx") as error:
        run(tmp_path, lambda *a, **k: pytest.fail(), files=[("old.xls", b"old")])
    assert "sensitive error" not in str(error.value)


def test_raw_text_excel_and_pdf_preserve_unknown_rows_and_locations():
    assert extract_evidence_units("a.txt", "标题\n陌生货 8\n只有规格".encode())[-1] == {"unit_id": "u3", "text": "只有规格", "line": 3}
    book = Workbook()
    book.active["B2"] = "陌生货"
    book.active["D2"] = 8
    book.create_sheet("第二页")["C3"] = "只有标题"
    buffer = BytesIO()
    book.save(buffer)
    units = extract_evidence_units("a.xlsx", buffer.getvalue())
    assert units[0]["text"] == "B=陌生货\tD=8" and units[0]["row"] == 2
    assert units[1]["sheet"] == "第二页" and units[1]["row"] == 3
    with fitz.open() as pdf:
        pdf.new_page().insert_text((30, 40), "unknown product 8")
        units = extract_evidence_units("a.pdf", pdf.tobytes())
    assert "unknown product 8" in units[0]["text"] and units[0]["page"] == 1 and units[0]["bbox"]


def test_local_ocr_raw_blocks_preserved_and_partial_invalid_result_rejected():
    image = BytesIO()
    Image.new("RGB", (20, 20), "white").save(image, "PNG")
    raw = [{"res": {"parsing_res_list": [{"block_label": "text", "block_content": "陌生品名", "block_bbox": [1, 2, 3, 4], "block_order": 1},
                                        {"block_label": "table", "block_content": "<table>未知表格</table>"}]}}]
    ocr = SimpleNamespace(extract_raw=lambda data: raw)
    units = extract_evidence_units("scan.png", image.getvalue(), ocr)
    assert len(units) == 2 and units[0]["bbox"] == [1, 2, 3, 4] and units[1]["block_label"] == "table"
    raw.append({"unexpected": "nonempty lost page"})
    with pytest.raises(ValueError, match="结构异常"):
        extract_evidence_units("scan.png", image.getvalue(), ocr)


def test_mixed_pdf_keeps_text_and_one_local_ocr_reading_of_same_page():
    image = BytesIO()
    Image.new("RGB", (20, 20), "white").save(image, "PNG")
    calls = []
    def extract_raw(data):
        calls.append(data)
        return [{"res": {"parsing_res_list": [{"block_label": "text", "block_content": "图片里的陌生货"}]}}]
    with fitz.open() as pdf:
        page = pdf.new_page()
        page.insert_text((30, 40), "inquiry title")
        page.insert_image(fitz.Rect(30, 60, 100, 130), stream=image.getvalue())
        units = extract_evidence_units("mixed.pdf", pdf.tobytes(), SimpleNamespace(extract_raw=extract_raw))
    assert len(calls) == 1 and len(units) == 2
    assert {unit["reading_method"] for unit in units} == {"pdf_text", "local_ocr"}
    assert {unit["page"] for unit in units} == {1}
    assert "inquiry title" in units[0]["text"] and units[1]["text"] == "图片里的陌生货"


def test_ui_button_calls_agent_then_requires_human_confirmation(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-secret")
    monkeypatch.setattr(workflow, "stream_transport", model_transport(calls))
    monkeypatch.setattr(inquiry, "RUNS_DIR", tmp_path)
    app = AppTest.from_file(str(inquiry.ROOT / "app.py"), default_timeout=20).run()
    app.checkbox(key="external_consent").check().run()
    assert not app.exception
    app.text_area[0].set_value(TEXT)
    next(button for button in app.button if button.label == "识别新询价").click().run()
    assert not app.exception and len(calls) == 2
    assert app.session_state["recognized"] and not app.session_state["records"][0].manually_confirmed
    assert list(app.dataframe[0].value.columns) == [
        "category", "product_name", "specification", "quantity_cartons", "price_marker",
    ]
    assert not any(button.label == "匹配标准主数据" for button in app.button)
    next(button for button in app.button if button.label == "确认识别结果").click().run()
    assert not app.exception and app.session_state["recognition_confirmed"]
    next(button for button in app.button if button.label == "匹配标准主数据").click().run()
    assert not app.exception and app.session_state["matched"]
    assert app.session_state["records"][0].unit_price is not None


@pytest.mark.parametrize("outcome", ["success", "input_failure", "budget_limit", "network_failure"])
def test_ui_status_tracks_real_stages_and_keeps_terminal_state(tmp_path, monkeypatch, outcome):
    import streamlit as st
    from streamlit.elements.lib.mutable_status_container import StatusContainer
    from src.ocr_pipeline import LocalOCRExtractor

    image = BytesIO()
    Image.new("RGB", (24, 24), "white").save(image, "PNG")
    upload = SimpleNamespace(name="synthetic.png", type="image/png", getvalue=image.getvalue)
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: [upload])
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-secret")
    monkeypatch.setattr(inquiry, "RUNS_DIR", tmp_path)
    stages, calls = [], []
    original_update = StatusContainer.update
    def update(status, **changes):
        stages.append((changes.get("label"), changes.get("state", "running")))
        return original_update(status, **changes)
    monkeypatch.setattr(StatusContainer, "update", update)
    def extract_raw(self, data):
        assert stages[-1] == ("正在识别图片…", "running")
        if outcome == "input_failure":
            raise ValueError("sensitive local error")
        return [{"res": {"parsing_res_list": [{"block_content": TEXT}]}}]
    monkeypatch.setattr(LocalOCRExtractor, "extract_raw", extract_raw)
    delegate = model_transport(calls)
    def transport(*args, **kwargs):
        assert stages[-1] == ("模型正在推理…", "running")
        if outcome == "network_failure":
            calls.append(1)
            raise workflow.TransportError("HTTP_402")
        return delegate(*args, **kwargs)
    monkeypatch.setattr(workflow, "stream_transport", transport)
    if outcome == "budget_limit":
        monkeypatch.setattr(inquiry, "RUN_CONFIG", {**inquiry.RUN_CONFIG, "budget": "0.01"})
    app = AppTest.from_file(str(inquiry.ROOT / "app.py"), default_timeout=20).run()
    app.checkbox(key="external_consent").check().run()
    next(button for button in app.button if button.label == "识别新询价").click().run()
    assert not app.exception
    assert app.status[0].state == ("complete" if outcome == "success" else "error")
    assert stages[0:2] == [("正在读取询价文件…", "running"), ("正在识别图片…", "running")]
    assert stages[-1][1] == app.status[0].state
    if outcome in {"input_failure", "budget_limit"}:
        assert not calls and "模型正在推理…" not in [label for label, state in stages]
    else:
        assert len(calls) == (2 if outcome == "success" else 1)
    if outcome == "input_failure":
        assert "recognized" not in app.session_state
        assert all("sensitive" not in error.value for error in app.error)
    elif outcome == "success":
        assert not app.session_state["records"][0].manually_confirmed


@pytest.mark.parametrize("provider_output", [None, '{"coverage":[1]}', '"invalid shape"'])
def test_ui_provider_failure_is_visible_without_rules_fallback(tmp_path, monkeypatch, provider_output):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-secret")
    def failed(*args, **kwargs):
        if provider_output is not None:
            return response(provider_output)
        raise workflow.TransportError("HTTP_402")
    monkeypatch.setattr(workflow, "stream_transport", failed)
    monkeypatch.setattr(inquiry, "RUNS_DIR", tmp_path)
    app = AppTest.from_file(str(inquiry.ROOT / "app.py"), default_timeout=20).run()
    app.checkbox(key="external_consent").check().run()
    app.text_area[0].set_value(TEXT)
    next(button for button in app.button if button.label == "识别新询价").click().run()
    assert not app.exception and app.session_state["records"][0].product_name is None
    assert any("模型未完成有效识别" in warning.value for warning in app.warning)
    next(button for button in app.button if button.label == "确认识别结果").click().run()
    assert app.error and not any(button.label == "匹配标准主数据" for button in app.button)
    pending = app.session_state["records"][0]
    pending.category, pending.product_name, pending.specification, pending.quantity_cartons = "沟槽", "90°弯头", "108", 1
    next(button for button in app.button if button.label == "确认识别结果").click().run()
    assert not app.exception and any("仅填写五列不能确认" in error.value for error in app.error)
    assert not pending.manually_confirmed and pending.price_marker is None
    assert not any(button.label == "匹配标准主数据" for button in app.button)


@pytest.mark.parametrize("marker", ["A", "a", "b"])
def test_ui_unknown_grade_requires_selection_then_matches_selected_tier(monkeypatch, marker):
    record = synthetic_record(0, "陌生档")
    record.category = "玛钢"
    record.product_name, record.specification = "弯头", "65"
    app = ui_records(monkeypatch, [record])
    edit_recognition(app, edited={0: {"price_marker": marker}})
    edit_recognition(app, confirm=True)
    assert not app.exception and app.session_state["recognition_confirmed"]
    confirmed = app.session_state["records"][0]
    assert confirmed.original_grade_text == "陌生档"
    assert confirmed.confirmed_inquiry[-1] == marker
    next(button for button in app.button if button.label == "匹配标准主数据").click().run()
    matched = app.session_state["records"][0]
    assert not app.exception and matched.price_marker == marker
    assert matched.match_status == "matched" and matched.unit_price is not None


def test_ui_confirmed_grade_edit_invalidates_match_and_preserves_selection(monkeypatch):
    app = ui_records(monkeypatch, [synthetic_record(0, "a")])
    next(button for button in app.button if button.label == "确认识别结果").click().run()
    next(button for button in app.button if button.label == "匹配标准主数据").click().run()
    app.session_state["generated"] = ("stale.pdf", {})
    edit_recognition(app, edited={0: {"price_marker": "b"}})
    assert not app.exception
    assert not app.session_state["recognition_confirmed"] and not app.session_state["matched"]
    assert "generated" not in app.session_state
    edit_recognition(app, confirm=True)
    assert not app.exception and app.session_state["recognition_confirmed"]
    assert app.session_state["records"][0].price_marker == "b"


def ui_records(monkeypatch, records):
    monkeypatch.setattr(workflow, "stream_transport", lambda *a, **k: pytest.fail("UI edits must not call a model"))
    app = AppTest.from_file(str(inquiry.ROOT / "app.py"), default_timeout=20).run()
    app.checkbox(key="external_consent").check().run()
    app.session_state["records"] = records
    app.session_state["recognized"] = True
    return app.run()


def edit_recognition(app, *, deleted=None, added=None, edited=None, confirm=False):
    if confirm:
        next(button for button in app.button if button.label == "确认识别结果").click()
    # AppTest has no public data-editor control; send its actual browser JSON state.
    states = app._tree.get_widget_states()
    editor = app.dataframe[0]
    delta = app.session_state[editor.key] if deleted is added is edited is None else {
        "deleted_rows": deleted or [], "added_rows": added or [], "edited_rows": edited or {}}
    states.widgets.add(id=editor.proto.id, string_value=json.dumps(delta))
    return app._run(widget_state=states)


def synthetic_record(index, grade="b", **changes):
    return ProductRecord("text", f"source-{index}.txt", source_location=f"line-{index}",
                         original_text=f"evidence-{index}", original_product_name=f"raw-name-{index}",
                         original_specification=f"raw-spec-{index}", original_grade_text=grade,
                         price_marker=grade if grade in {"a", "A", "b"} else None,
                         category="沟槽", product_name="90°弯头", specification="108", quantity_cartons=1,
                         confidence=0.8, warnings=[f"warning-{index}"], **changes)


@pytest.mark.parametrize("deleted", [0, 1, 2])
@pytest.mark.parametrize("confirmed", [False, True])
def test_ui_delete_add_preserves_evidence_and_does_not_reapply_edits(monkeypatch, deleted, confirmed):
    originals = [synthetic_record(index, grade) for index, grade in enumerate(("a", "A", "b"))]
    app = ui_records(monkeypatch, originals)
    if confirmed:
        next(button for button in app.button if button.label == "确认识别结果").click().run()
    new_row = {"category": "沟槽", "product_name": "90°弯头", "specification": "108", "quantity_cartons": 2}
    edit_recognition(app, deleted=[deleted], added=[new_row], confirm=True)
    assert not app.exception and app.session_state["recognition_confirmed"]
    expected = [record for index, record in enumerate(originals) if index != deleted]
    for _ in range(2):
        current = app.session_state["records"]
        assert len(current) == 3
        for actual, source in zip(current[:2], expected):
            for name in ("source_name", "source_location", "original_text", "original_product_name",
                         "original_specification", "original_grade_text", "price_marker", "confidence", "warnings"):
                assert getattr(actual, name) == getattr(source, name)
        assert current[2].source_type == "manual" and current[2].original_text == ""
        assert current[2].original_grade_text is None and current[2].price_marker == "b"
        assert current[2].quantity_cartons == 2 and current[2].manually_confirmed
        edit_recognition(app)
        assert not app.exception and app.session_state["recognition_confirmed"]


@pytest.mark.parametrize("blocked", ["grade", "placeholder"])
def test_ui_delete_does_not_detach_unknown_grade_or_pending_gate(monkeypatch, blocked):
    pending = synthetic_record(2, "陌生档" if blocked == "grade" else "",
                               extraction_method="待人工处理" if blocked == "placeholder" else "AI")
    app = ui_records(monkeypatch, [synthetic_record(0), synthetic_record(1), pending])
    edit_recognition(app, deleted=[0])
    for _ in range(2):
        edit_recognition(app, confirm=True)
        assert not app.exception and not app.session_state.get("recognition_confirmed")
        message = "等级/a-A-b 有歧义" if blocked == "grade" else "仅填写五列不能确认"
        assert any(message in error.value for error in app.error)
        assert not any(button.label == "匹配标准主数据" for button in app.button)


def test_ui_same_values_replacement_invalidates_confirmation_and_generated(monkeypatch):
    app = ui_records(monkeypatch, [synthetic_record(0), synthetic_record(1)])
    next(button for button in app.button if button.label == "确认识别结果").click().run()
    next(button for button in app.button if button.label == "匹配标准主数据").click().run()
    assert app.session_state["matched"]
    app.session_state["generated"] = ("stale.pdf", {})
    same_row = app.dataframe[0].value.iloc[-1].to_dict()
    edit_recognition(app, deleted=[1], added=[same_row])
    edit_recognition(app)
    assert not app.exception and not app.session_state["recognition_confirmed"]
    assert not app.session_state["matched"] and "generated" not in app.session_state
    assert not any(button.label == "匹配标准主数据" for button in app.button)
    assert all(record.unit_price is None and not record.manually_confirmed for record in app.session_state["records"])
    edit_recognition(app, confirm=True)
    assert not app.exception and app.session_state["recognition_confirmed"]
    assert app.session_state["records"][-1].source_type == "manual"


def test_ui_duplicate_upload_names_have_distinct_download_keys(monkeypatch):
    import streamlit as st

    uploads = [SimpleNamespace(name="同名询价.txt", type="text/plain", getvalue=lambda: b"first"),
               SimpleNamespace(name="同名询价.txt", type="text/plain", getvalue=lambda: b"second")]
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: uploads)
    monkeypatch.setattr(workflow, "stream_transport", lambda *a, **k: pytest.fail("Preview must not call a model"))
    app = AppTest.from_file(str(inquiry.ROOT / "app.py"), default_timeout=20).run()
    app.checkbox(key="external_consent").check().run()
    assert not app.exception
    assert len(app.get("download_button")) == 2
    assert len({button.proto.id for button in app.get("download_button")}) == 2


@pytest.mark.parametrize("field", ["product_name", "specification"])
def test_ui_whitespace_only_fields_keep_draft_and_report_error(monkeypatch, field):
    app = ui_records(monkeypatch, [synthetic_record(0)])
    edit_recognition(app, edited={0: {field: " \t "}}, confirm=True)
    assert not app.exception and not app.session_state.get("recognition_confirmed")
    assert any("不能为空" in error.value or "缺少" in error.value for error in app.error)
    assert app.session_state[app.dataframe[0].key]["edited_rows"][0][field] == " \t "
    assert not any(button.label == "匹配标准主数据" for button in app.button)


@pytest.mark.parametrize("file_kind", ["pdf", "png"])
@pytest.mark.parametrize("failure", ["missing", "unreadable"])
def test_ui_unreadable_generated_files_clear_downloads_but_keep_inquiry(tmp_path, monkeypatch, file_kind, failure):
    from src import core

    monkeypatch.setattr(core, "generate_quotation", lambda *a, **k: pytest.fail("Do not regenerate automatically"))
    app = ui_records(monkeypatch, [synthetic_record(0)])
    next(button for button in app.button if button.label == "确认识别结果").click().run()
    next(button for button in app.button if button.label == "匹配标准主数据").click().run()
    pdf, png = tmp_path / "synthetic.pdf", tmp_path / "synthetic.png"
    pdf.write_bytes(b"synthetic pdf")
    png.write_bytes(b"synthetic png")
    target = pdf if file_kind == "pdf" else png
    if failure == "missing":
        target.unlink()
    else:
        read_bytes = Path.read_bytes
        def unreadable(path):
            if path == target:
                raise PermissionError("synthetic unreadable file")
            return read_bytes(path)
        monkeypatch.setattr(Path, "read_bytes", unreadable)
    app.session_state["generated"] = (pdf, {
        "sales_id": "QA-SYNTH-TESTS", "page_count": 1, "total_label": "合计", "total": 0, "png_paths": [png],
    })
    app.run()
    assert not app.exception
    assert any("输出文件不可读取，请重新生成" in error.value for error in app.error)
    assert "generated" not in app.session_state
    assert app.session_state["recognition_confirmed"] and app.session_state["matched"]
    assert app.session_state["records"][0].original_text == "evidence-0"
    assert not app.get("download_button")
    assert not any("已生成销单" in success.value for success in app.success)
    assert (png if file_kind == "pdf" else pdf).exists()
