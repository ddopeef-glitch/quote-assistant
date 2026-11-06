"""Offline protocol and bounded-run regression checks; never calls a provider."""
import copy
import http.client
import json
from pathlib import Path
from types import SimpleNamespace
import urllib.error

import pytest

from src import agent_workflow as workflow


def event(delta=None, finish=None, usage=None):
    value = {"model": "deepseek-v4-pro", "choices": [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if usage is not None:
        value["usage"] = usage
    return "data: " + json.dumps(value, ensure_ascii=False) + "\n\n"


def answer(data, *, unresolved=False):
    return {"document_type": "unknown", "items": [], "coverage": [{"source_id": data["document"]["source_id"],
        "unit_id": "u1", "disposition": "unresolved" if unresolved else "non_inquiry", "item_ids": [],
        "reason": "需人工判读" if unresolved else "非询价"}], "unresolved": ["需人工判读"] if unresolved else []}


def response(content="", calls=None, finish=None):
    message = {"role": "assistant", "content": content, "reasoning_content": "保留推理片段"}
    if calls:
        message["tool_calls"] = calls
    return {"model": "deepseek-v4-pro", "usage": {"prompt_tokens": 20, "completion_tokens": 10,
        "total_tokens": 30, "prompt_cache_hit_tokens": 5, "prompt_cache_miss_tokens": 15},
        "finish_reason": finish or ("tool_calls" if calls else "stop"), "message": message}


def tool_call(ident="call_1", query_id="q1", specification=None):
    return {"id": ident, "type": "function", "function": {"name": "query_batch", "arguments": json.dumps({"queries": [
        {"query_id": query_id, "product_terms": ["弯头"], "specifications": [] if specification is None else [specification], "categories": [], "price_markers": []}]}, ensure_ascii=False)}}


class FakeCatalog:
    catalog_fingerprint = "frozen-test-catalog"

    def index_payload(self):
        return [{"product_name": "弯头", "aliases": []}]

    def query_batch(self, queries):
        return {"results": [{"query_id": q["query_id"], "candidates": [{"catalog_id": "cat_one"}],
            "total_matches": 1, "has_more": False} for q in queries]}

    def validate_proposal(self, items, *, allowed_catalog_ids):
        return {"valid": True, "errors": [], "unresolved": [], "quote_authorized": False}


@pytest.fixture
def setup(monkeypatch):
    from src import agent_catalog
    monkeypatch.setattr(agent_catalog.CatalogSnapshot, "from_eval_catalog", classmethod(lambda cls, payload: FakeCatalog()))
    data = {"schema_version": "test", "document": {"source_id": "s1", "images": [], "units": [{"unit_id": "u1", "text": "保留陌生词"}]},
            "catalog": {"products": [{"catalog_id": "cat_one", "category": "沟槽", "product_name": "弯头", "specification": "25",
                "price_marker": "b", "aliases": [], "status": "启用"}], "grade_mappings": [{"raw": "中", "price_marker": "a"}]}}
    config = {"max_requests": 72, "max_calls_per_sample": 4, "total_seconds": 14400, "sample_seconds": 1800,
              "request_seconds": 1200, "idle_seconds": 120, "input_per_million": "9", "output_per_million": "27", "currency": "CNY", "budget": None}
    return data, config


def run(tmp_path, setup, transport, **kwargs):
    data, config = setup
    return workflow.run_ab(config, [("sample", "file_hash", data)], "共同 JSON 规则", tmp_path / "run", execute=True,
                           transport=transport, environ={"DEEPSEEK_API_KEY": "secret-test-value"}, **kwargs)


def test_sse_fragmented_tools_and_reasoning():
    parser = workflow.SSEParser()
    stream = event({"reasoning_content": "先查"})
    stream += event({"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "query_", "arguments": '{"queries":'}}]})
    stream += event({"tool_calls": [{"index": 0, "function": {"name": "batch", "arguments": "[]}"}}]}, "tool_calls")
    stream += event(usage={"prompt_tokens": 7, "completion_tokens": 8}) + "data: [DONE]\n\n"
    for character in stream:
        parser.feed(character)
    result = parser.finish()
    assert result["message"]["reasoning_content"] == "先查"
    assert result["message"]["tool_calls"][0]["function"] == {"name": "query_batch", "arguments": '{"queries":[]}'}
    assert result["usage"]["completion_tokens"] == 8


@pytest.mark.parametrize("stream,code", [
    (event({"content": "{}"}, "stop"), "truncated_sse"),
    (event({"content": "{}"}) + "data: [DONE]\n\n", "missing_stream_metadata"),
    (event({"content": "{}"}, "tool_calls") + "data: [DONE]\n\n", "tool_finish_mismatch"),
    ('data: {"model":"x","model":"y"}\n\n', "invalid_sse_json"),
])
def test_sse_rejects_incomplete_or_invalid(stream, code):
    parser = workflow.SSEParser()
    with pytest.raises(workflow.EvalError, match=code):
        parser.feed(stream)
        parser.finish()


def test_dry_run_has_no_credentials_or_output(tmp_path, setup):
    data, config = setup
    report = workflow.run_ab(config, [("sample", "hash", data)], "JSON", tmp_path / "absent")
    assert report["mode"] == "dry_run"
    assert not (tmp_path / "absent").exists()


def test_two_round_query_replays_real_tool_protocol_and_equal_parameters(tmp_path, setup):
    data, _ = setup
    payloads = []
    def transport(payload, key, output, **limits):
        payloads.append(copy.deepcopy(payload))
        if "tools" in payload and len(payload["messages"]) == 2:
            return response(calls=[tool_call()])
        return response(workflow.dumps(answer(data, unresolved=True)))
    report = run(tmp_path, setup, transport)
    assert [s["model_requests"] for s in report["samples"]] == [1, 2]
    assert all(p["max_tokens"] == 65536 and p["thinking"] == {"type": "enabled"} and p["reasoning_effort"] == "high" for p in payloads)
    assert all("temperature" not in p and "response_format" not in p for p in payloads)
    followup = payloads[-1]["messages"]
    assert followup[2]["role"] == "assistant" and followup[2]["reasoning_content"] == "保留推理片段"
    assert followup[3]["role"] == "tool" and followup[3]["tool_call_id"] == "call_1"
    assert "products" not in json.loads(payloads[1]["messages"][1]["content"])["catalog"]
    assert report["samples"][1]["allowed_catalog_ids"] == ["cat_one"]
    assert report["samples"][1]["status"] == "unresolved"
    assert report["known_usage_totals"]["prompt_cache_hit_tokens"] == 15


def test_network_failure_retried_once_and_counted(tmp_path, setup):
    calls = []
    def transport(payload, key, output, **limits):
        calls.append(payload)
        raise workflow.TransportError("network_timeout", retryable=True)
    report = run(tmp_path, setup, transport)
    assert len(calls) == report["all_reserved_requests"] == 2
    assert report["requests_without_usage"] == 2
    assert report["samples"][0]["status"] == "request_failed"
    assert report["samples"][0]["retry_stop_reason"] == "network_retry_limit"
    assert [row["network_retry"] for row in report["samples"][0]["rounds"]] == [False, True]
    assert report["samples"][0]["rounds"][0]["request_sha256"] == report["samples"][0]["rounds"][1]["request_sha256"]


def test_insufficient_balance_stops_without_retry_or_other_arm(tmp_path, setup):
    calls = []
    def transport(*args, **kwargs):
        calls.append(1)
        raise workflow.TransportError("HTTP_402")
    result = run(tmp_path, setup, transport)
    assert len(calls) == 1 and len(result["samples"]) == 1


def test_call_limit_includes_network_retry(tmp_path, setup):
    data, config = setup
    calls = []
    def transport(payload, key, output, **limits):
        if "tools" not in payload:
            return response(workflow.dumps(answer(data)))
        calls.append(1)
        if len(calls) == 2:
            raise workflow.TransportError("network_timeout", retryable=True)
        return response(calls=[tool_call("call_" + str(len(calls)), "q" + str(len(calls)), str(len(calls)))])
    result = run(tmp_path, setup, transport)
    assert len(calls) == 4
    assert result["samples"][-1]["status"] == "call_limit"


def test_repeated_query_stops_without_third_query(tmp_path, setup):
    data, _ = setup
    count = []
    def transport(payload, key, output, **limits):
        if "tools" not in payload:
            return response(workflow.dumps(answer(data)))
        count.append(1)
        return response(calls=[tool_call("call_" + str(len(count)), "q" + str(len(count)))])
    report = run(tmp_path, setup, transport)
    assert len(count) == 2
    assert report["samples"][-1]["status"] == "no_new_information"


def test_raw_json_is_not_cleaned_or_repaired(tmp_path, setup):
    def transport(*args, **kwargs):
        return response('```json\n{}\n```')
    report = run(tmp_path, setup, transport)
    assert all(s["status"] == "structure_failed" and s["model_requests"] == 1 for s in report["samples"])
    saved = json.loads((tmp_path / "run/sample__A/round-01/response.json").read_text(encoding="utf-8"))
    assert saved["message"]["content"].startswith("```json")


def test_resume_skips_evidence_and_preserves_request_count(tmp_path, setup):
    data, _ = setup
    run(tmp_path, setup, lambda *args, **kwargs: response(workflow.dumps(answer(data))))
    before = {p: p.read_bytes() for p in (tmp_path / "run/sample__A").rglob("*") if p.is_file()}
    report = run(tmp_path, setup, lambda *args, **kwargs: pytest.fail("must not call on resume"), resume=True)
    assert report["all_reserved_requests"] == 2
    assert all(p.read_bytes() == raw for p, raw in before.items())


def test_resume_rejects_version_change_and_partial_evidence(tmp_path, setup):
    data, _ = setup
    run(tmp_path, setup, lambda *args, **kwargs: response(workflow.dumps(answer(data))))
    with pytest.raises(workflow.EvalError, match="fingerprint"):
        run(tmp_path, setup, lambda *args, **kwargs: None, resume=True, source_hashes={"code": "changed"})
    (tmp_path / "run/sample__A/summary.json").unlink()
    with pytest.raises(workflow.EvalError, match="incomplete"):
        run(tmp_path, setup, lambda *args, **kwargs: None, resume=True)


def test_secret_redaction_and_safe_exception_text(tmp_path, setup):
    def transport(*args, **kwargs):
        raise RuntimeError("secret-test-value server echoed header")
    run(tmp_path, setup, transport)
    assert all("secret-test-value" not in p.read_text(encoding="utf-8") for p in (tmp_path / "run").rglob("*") if p.is_file())


def test_sample_time_limit_blocks_next_paid_round(tmp_path, setup, monkeypatch):
    data, config = setup
    config["sample_seconds"] = 2
    clock = [0.0]
    monkeypatch.setattr(workflow.time, "monotonic", lambda: clock[0])
    b_calls = []
    def transport(payload, key, output, **limits):
        if "tools" not in payload:
            return response(workflow.dumps(answer(data)))
        b_calls.append(1)
        assert limits["timeout_seconds"] <= 2
        clock[0] += 3
        return response(calls=[tool_call()])
    report = run(tmp_path, setup, transport)
    assert len(b_calls) == 1 and report["samples"][-1]["status"] == "sample_timeout"


@pytest.mark.parametrize("field", ["max_requests", "max_calls_per_sample", "total_seconds", "sample_seconds", "request_seconds"])
def test_legacy_evaluation_still_requires_finite_limits(tmp_path, setup, field):
    setup[1][field] = None
    with pytest.raises(workflow.EvalError, match="invalid " + field):
        run(tmp_path, setup, lambda *a, **k: pytest.fail("configuration must block transport"))


def test_legacy_evaluation_keeps_shared_request_limit(tmp_path, setup):
    data, config = setup
    config["max_requests"] = 1
    calls = []
    def transport(*args, **kwargs):
        calls.append(1)
        return response(workflow.dumps(answer(data)))
    report = run(tmp_path, setup, transport)
    assert len(calls) == report["all_reserved_requests"] == 1
    assert len(report["samples"]) == 1 and report["samples"][0]["arm"] == "A"


def test_alternating_order_stable_for_subset(tmp_path, setup):
    data, config = setup
    entries = [("one", "h1", data), ("two", "h2", data), ("three", "h3", data)]
    plan = workflow.run_ab(config, entries, "JSON", tmp_path / "out", subset=["two", "three"])
    assert plan["schedule"] == [("two", "B"), ("two", "A"), ("three", "A"), ("three", "B")]


@pytest.mark.parametrize("timeout_seconds", [None, 5])
def test_stream_transport_saves_raw_and_usage_redacts_key(tmp_path, monkeypatch, timeout_seconds):
    raw = (event({"content": '{"message":"secret-test-value"}', "reasoning_content": "合成片段"}, "stop") +
           event(usage={"prompt_tokens": 1, "completion_tokens": 2}) + "data: [DONE]\n\n").encode("utf-8")
    chunks = [bytes([value]) for value in raw] + [b""]
    timeouts = []
    fake_response = SimpleNamespace(status=200, fp=SimpleNamespace(raw=SimpleNamespace(_sock=SimpleNamespace(settimeout=timeouts.append))),
        getheader=lambda _: "text/event-stream", read1=lambda _: chunks.pop(0), close=lambda: None)
    fake_response.getheader = lambda *args: "text/event-stream"
    def open_http(*args, **kwargs):
        timeouts.append(kwargs["timeout"])
        return fake_response
    monkeypatch.setattr(workflow.urllib.request, "build_opener", lambda *args: SimpleNamespace(open=open_http))
    result = workflow.stream_transport({"messages": []}, "secret-test-value", tmp_path, timeout_seconds=timeout_seconds, idle_seconds=1)
    assert result["usage"] == {"prompt_tokens": 1, "completion_tokens": 2}
    assert timeouts and all(0 < value <= 1 for value in timeouts)
    assert result["message"]["reasoning_content"] == "合成片段"
    assert "secret-test-value" not in (tmp_path / "raw.sse.txt").read_text(encoding="utf-8")


def test_redirect_refused():
    with pytest.raises(workflow.EvalError, match="redirect refused"):
        workflow.NoRedirect().redirect_request(None, None, 302, "", {}, "https://other.example/")


@pytest.mark.parametrize("value", ["NaN", '{"x":1,"x":2}', '{"document_type":[],"items":[{}],"coverage":[],"unresolved":[]}'])
def test_malformed_answers_remain_structural_failures(value, setup):
    data, _ = setup
    result = workflow.check_answer(value, data, FakeCatalog(), set())
    assert result["status"] == "structure_failed"


@pytest.mark.parametrize("failure", ["name", "arguments", "query"])
def test_model_tool_errors_are_not_network_failures(tmp_path, setup, failure, monkeypatch):
    data, _ = setup
    def transport(payload, key, output, **limits):
        if "tools" not in payload:
            return response(workflow.dumps(answer(data)))
        call = tool_call()
        if failure == "name":
            call["function"]["name"] = "shell"
        if failure == "arguments":
            call["function"]["arguments"] = "broken-json"
        return response(calls=[call])
    if failure == "query":
        monkeypatch.setattr(FakeCatalog, "query_batch", lambda *args: (_ for _ in ()).throw(ValueError("bad query")))
    report = run(tmp_path, setup, transport)
    row = report["samples"][-1]
    assert row["status"] == ("tool_failed" if failure == "query" else "protocol_failed")
    assert row["model_requests"] == 1 and row["rounds"][0]["usage"]["completion_tokens"] == 10


def test_real_catalog_validation_integration_keeps_unknown_and_rejects_mismatch():
    from src.agent_catalog import CatalogSnapshot
    catalog = {"products": [{"catalog_id": "cat_0000000000000001", "category": "玛钢", "product_name": "90°弯头", "specification": "25",
        "price_marker": "b", "aliases": ["弯头"], "status": "启用"}], "grade_mappings": [{"raw": "中", "price_marker": "a"}]}
    snapshot = CatalogSnapshot.from_eval_catalog(catalog)
    data = {"catalog": catalog, "document": {"source_id": "s1", "units": [{"unit_id": "u1", "text": "25弯头X10件"}]}}
    item = {"item_id": "i1", "source_id": "s1", "evidence": [{"unit_id": "u1", "quote": "25弯头X10件"}],
            "product_name_raw": "弯头", "specification_raw": "25", "quantity_raw": "10件", "grade_raw": "X", "category": "玛钢",
            "product_name": "90°弯头", "specification": "25", "quantity_cartons": 10, "price_marker": None,
            "candidate_ids": [], "selected_candidate_id": None, "unresolved": ["未知档位X"]}
    value = {"document_type": "inquiry", "items": [item], "coverage": [{"source_id": "s1", "unit_id": "u1", "disposition": "items", "item_ids": ["i1"], "reason": "保留原文"}], "unresolved": []}
    result = workflow.check_answer(workflow.dumps(value), data, snapshot, set())
    assert result["status"] == "unresolved" and not result["business_errors"]
    assert result["parsed"]["items"][0]["specification"] == "25"
    item.update(price_marker="b", selected_candidate_id="cat_0000000000000001", candidate_ids=["cat_0000000000000001"])
    result = workflow.check_answer(workflow.dumps(value), data, snapshot, {"cat_0000000000000001"})
    assert result["status"] == "business_failed"


def test_resume_extends_subset_without_duplicate_paid_calls(tmp_path, setup):
    data, config = setup
    entries = [("one", "h1", data), ("two", "h2", data)]
    calls = []
    def transport(payload, key, output, **limits):
        calls.append(output)
        return response(workflow.dumps(answer(data)))
    options = dict(execute=True, transport=transport, environ={"DEEPSEEK_API_KEY": "test"})
    workflow.run_ab(config, entries, "JSON", tmp_path / "out", subset=["one"], **options)
    report = workflow.run_ab(config, entries, "JSON", tmp_path / "out", resume=True, **options)
    assert len(calls) == 4 and report["all_reserved_requests"] == 4
    assert report["known_usage_totals"]["total_tokens"] == 120


@pytest.mark.parametrize("changed_when", ["before", "after"])
def test_master_change_stops_and_seals_run(tmp_path, setup, changed_when):
    data, _ = setup
    checks, calls = [], []
    def assert_current():
        checks.append(1)
        if changed_when == "before" or len(checks) >= 2:
            raise ValueError("changed")
    master = SimpleNamespace(master_fingerprint="master-hash", assert_current=assert_current)
    def transport(*args, **kwargs):
        calls.append(1)
        return response(workflow.dumps(answer(data)))
    result = run(tmp_path, setup, transport, master_snapshot=master)
    assert result["samples"][0]["status"] == "stale_source"
    assert len(calls) == (0 if changed_when == "before" else 1)
    if calls:
        assert result["samples"][0]["rounds"][0]["usage"]["completion_tokens"] == 10
    with pytest.raises(workflow.EvalError, match="closed"):
        run(tmp_path, setup, transport, master_snapshot=master, resume=True)


@pytest.mark.parametrize("fault", ["finish_conflict", "missing_metadata", "tool_index", "tool_id", "truncated",
                                  "truncated_json", "truncated_event", "truncated_done", "truncated_utf8",
                                  "incomplete_read", "invalid_json", "invalid_encoding"])
def test_http_sse_failure_source_preserves_evidence_and_bounds_retries(tmp_path, setup, monkeypatch, fault):
    """Exercise the real parser and transport, replacing only the HTTP connection."""
    usage = {"prompt_tokens": 12, "completion_tokens": 9, "total_tokens": 21}
    tool = {"index": 1 if fault == "tool_index" else 0, "id": "" if fault == "tool_id" else "call_1",
            "type": "function", "function": {"name": "query_batch", "arguments": '{"queries":[]}'}}
    network_fault = fault.startswith("truncated") or fault == "incomplete_read"
    finish = "stop" if fault == "finish_conflict" or network_fault else None if fault == "missing_metadata" else "tool_calls"
    delta = {"content": "{}"} if fault == "missing_metadata" or network_fault else {"tool_calls": [tool]}
    prefix = (event(delta, finish) + event(usage=usage)).encode("utf-8")
    tail = {"truncated": b"", "truncated_json": b'data: {"choices":',
            "truncated_event": b'data: {"choices":[]}\n', "truncated_done": b"data: [DONE]\n",
            "truncated_utf8": b'data: {"text":"\xe4\xb8', "invalid_json": b"data: invalid\n\n",
            "invalid_encoding": b"data: \xff\n\n"}.get(fault, b"data: [DONE]\n\n")
    raw = prefix + tail
    http_calls = []
    def open_http(*args, **kwargs):
        http_calls.append(1)
        chunks = [prefix, tail, b""]
        def read_chunk(_):
            chunk = chunks.pop(0)
            if fault == "incomplete_read" and chunk == tail:
                raise http.client.IncompleteRead(chunk, 1)
            return chunk
        return SimpleNamespace(status=200,
            fp=SimpleNamespace(raw=SimpleNamespace(_sock=SimpleNamespace(settimeout=lambda _: None))),
            getheader=lambda *args: "text/event-stream", read1=read_chunk, close=lambda: None)
    monkeypatch.setattr(workflow.urllib.request, "build_opener", lambda *args: SimpleNamespace(open=open_http))
    master = SimpleNamespace(master_fingerprint="master-hash", assert_current=lambda: None)
    report = run(tmp_path, setup, workflow.stream_transport, master_snapshot=master)
    expected_status = "request_failed" if network_fault else "protocol_failed"
    expected_kind = "transport" if network_fault else "protocol"
    assert all(sample["status"] == expected_status for sample in report["samples"])
    # Full protocol errors do not retry; genuine truncation retries once and stops the batch.
    assert len(http_calls) == 2
    assert [sample["model_requests"] for sample in report["samples"]] == ([2] if network_fault else [1, 1])
    evidence = tmp_path / "run/sample__A/round-01"
    metadata = json.loads((evidence / "stream.json").read_text(encoding="utf-8"))
    expected_error = {"finish_conflict": "tool_finish_mismatch", "missing_metadata": "missing_stream_metadata",
                      "tool_index": "noncontiguous_tool_indexes", "tool_id": "invalid_complete_tools",
                      "incomplete_read": "incomplete_http_read", "invalid_json": "invalid_sse_json",
                      "invalid_encoding": "invalid_stream_encoding"}.get(fault, "truncated_sse")
    assert metadata["failure_kind"] == expected_kind and metadata["usage"] == usage
    assert metadata["error"] == expected_error
    assert metadata["finish_reason"] == finish
    assert metadata["done_received"] == (not network_fault and fault not in {"invalid_json", "invalid_encoding"})
    assert metadata["retryable"] is network_fault
    assert (evidence / "raw.sse.txt").read_text(encoding="utf-8") == raw.decode("utf-8", errors="replace")
    assert not (evidence / "response.json").exists() and not (evidence / "tool-results.json").exists()
    assert report["samples"][0]["rounds"][0]["usage"] == usage
    assert report["samples"][0]["rounds"][0]["error"] == expected_error
    ledger = [json.loads(line) for line in (tmp_path / "run/ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sum(row["event"] == "reserve" for row in ledger) == len(http_calls)
    results = [row for row in ledger if row["event"] == "result"]
    assert len(results) == len(http_calls) and all(row["status"] == expected_status and row["usage"] == usage for row in results)
