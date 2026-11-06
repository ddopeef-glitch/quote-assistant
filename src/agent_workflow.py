"""Bounded catalog proposals for inquiry recognition and offline experiments.

JSON/input/evidence helpers extracted from the frozen 2026-09-08 runner;
original files remain unchanged. Transport uses the official Chat Completions
SSE tool protocol, including reasoning_content replay for thinking tools.
"""
from __future__ import annotations

from collections import defaultdict
import codecs
from decimal import Decimal
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import socket
import time
import urllib.error
import urllib.request

ITEM_FIELDS = {"item_id", "source_id", "evidence", "product_name_raw", "specification_raw", "quantity_raw", "grade_raw", "category", "product_name", "specification", "quantity_cartons", "price_marker", "candidate_ids", "selected_candidate_id", "unresolved"}


class EvalError(ValueError):
    """Messages deliberately exclude request/response bodies and credentials."""


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def loads(text):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise EvalError("duplicate JSON key")
            result[key] = value
        return result
    try:
        return json.loads(text, object_pairs_hook=pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(EvalError("nonfinite JSON")))
    except (ValueError, TypeError, RecursionError):
        raise EvalError("invalid JSON") from None


def redact(text, secrets):
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return text


def save(path, value, secrets=()):
    # Exclusive creation preserves any previous experiment evidence.
    with path.open("x", encoding="utf-8") as handle:
        handle.write(redact(dumps(value), secrets) + "\n")


def reject_private(value, depth=0):
    if depth > 40:
        raise EvalError("input nesting limit")
    if isinstance(value, dict):
        for key, child in value.items():
            if key.startswith("_") or key.casefold() in {"gt", "ground_truth", "groundtruth", "api_key", "authorization"}:
                raise EvalError("private or ground-truth field in input")
            reject_private(child, depth + 1)
    elif isinstance(value, list):
        for child in value:
            reject_private(child, depth + 1)


def validate_answer(value, data):
    """Structural/evidence checks only; never treat them as semantic accuracy."""
    errors = []
    def require(condition, code):
        if not condition:
            errors.append(code)
    def strings(values):
        return isinstance(values, list) and all(isinstance(v, str) and v.strip() for v in values)
    if not isinstance(value, dict) or set(value) != {"document_type", "items", "coverage", "unresolved"}:
        return ["answer_fields"]
    require(value["document_type"] in ("inquiry", "price_matrix", "mixed", "unknown"), "document_type")
    require(strings(value["unresolved"]), "global_unresolved")
    if not isinstance(value["items"], list) or not isinstance(value["coverage"], list):
        return errors + ["items_or_coverage_type"]
    units = {u["unit_id"]: u["text"] for u in data["document"]["units"]}
    catalog = {p["catalog_id"] for p in data["catalog"]["products"]}
    source_id = data["document"]["source_id"]
    items, references = {}, defaultdict(set)
    for item in value["items"]:
        if not isinstance(item, dict) or set(item) != ITEM_FIELDS:
            errors.append("item_fields")
            continue
        ident = item["item_id"]
        if not isinstance(ident, str) or not ident.strip() or ident in items:
            errors.append("item_id")
            continue
        items[ident] = item
        require(item["source_id"] == source_id, "item_source")
        for field in ("product_name_raw", "specification_raw", "quantity_raw", "grade_raw", "product_name", "specification"):
            require(item[field] is None or isinstance(item[field], str) and bool(item[field].strip()), "item_text")
        require(item["category"] in (None, "沟槽", "玛钢", "球铁"), "category")
        require(item["price_marker"] in (None, "a", "A", "b"), "price_marker")
        quantity = item["quantity_cartons"]
        require(quantity is None or type(quantity) is int and quantity > 0, "quantity")
        require(strings(item["unresolved"]), "item_unresolved")
        candidates = item["candidate_ids"]
        if not strings(candidates):
            errors.append("candidate_ids")
            candidates = []
        require(len(candidates) == len(set(candidates)) and set(candidates) <= catalog, "unknown_or_duplicate_candidate")
        require(item["selected_candidate_id"] is None or item["selected_candidate_id"] in candidates, "selected_candidate")
        evidence = item["evidence"]
        if not isinstance(evidence, list) or not evidence:
            errors.append("evidence_empty")
            continue
        for citation in evidence:
            if not isinstance(citation, dict) or set(citation) != {"unit_id", "quote"}:
                errors.append("evidence_fields")
                continue
            unit_id, quote = citation["unit_id"], citation["quote"]
            if not isinstance(unit_id, str) or unit_id not in units:
                errors.append("evidence_unit")
                continue
            require(isinstance(quote, str) and bool(quote.strip()) and quote in units[unit_id], "evidence_quote")
            references[unit_id].add(ident)
    covered = set()
    for row in value["coverage"]:
        if not isinstance(row, dict) or set(row) != {"source_id", "unit_id", "disposition", "item_ids", "reason"}:
            errors.append("coverage_fields")
            continue
        unit_id = row["unit_id"]
        if not isinstance(unit_id, str) or unit_id not in units or unit_id in covered:
            errors.append("coverage_unit")
            continue
        covered.add(unit_id)
        require(row["source_id"] == source_id, "coverage_source")
        require(row["disposition"] in ("items", "context", "non_inquiry", "unresolved"), "coverage_disposition")
        require(isinstance(row["reason"], str) and bool(row["reason"].strip()), "coverage_reason")
        ids = row["item_ids"]
        if strings(ids):
            require(len(ids) == len(set(ids)) and set(ids) <= items.keys(), "coverage_items")
            require(set(ids) == references[unit_id], "coverage_evidence_consistency")
            require(row["disposition"] != "items" or bool(ids), "coverage_items_empty")
            require(row["disposition"] != "non_inquiry" or not ids, "non_inquiry_items")
        else:
            errors.append("coverage_items_type")
    require(covered == units.keys(), "coverage_incomplete")
    return sorted(set(errors))


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise EvalError("HTTP redirect refused")


ENDPOINT = "https://api.deepseek.com/chat/completions"
MODEL_PARAMETERS = {"model": "deepseek-flash", "max_tokens": 65536,
                    "thinking": {"type": "enabled"}, "reasoning_effort": "high",
                    "stream": True, "stream_options": {"include_usage": True}}
MAX_STREAM_BYTES = 64 * 1024 * 1024
MAX_REQUEST_BYTES = 8 * 1024 * 1024
ARM_INSTRUCTIONS = {
    "A": "目录提供方式：完整目录位于 catalog.products；直接比较目录并提交最终 JSON。",
    "B": "目录提供方式：初始 catalog 只有等级映射和去重品名/别名索引。理解完整原文后，"
         "通过 query_batch 一次批量提交各项检索假设，接收本地候选后提交最终 JSON。"
         "工具参数不是已确认事实；空数组表示不限制该条件。未知词仍须保留。"
         "有证据需扩大查询或分页时可再批量查询；重复相同查询不会产生新信息。"
         "通常查询及最终回答共两次。不能因无候选清空正确规格或删除原项。"
         "候选 ID 只能使用本轮工具实际返回的 ID。工具调用由 tools 协议返回，不能用普通 JSON 冒充。",
}


def sha256(value):
    raw = value if isinstance(value, bytes) else dumps(value).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class TransportError(EvalError):
    def __init__(self, code, *, retryable=False, metadata=None, failure_kind="transport"):
        super().__init__(code)
        self.retryable = retryable
        self.metadata = metadata or {}
        self.failure_kind = failure_kind


class ProtocolError(TransportError):
    def __init__(self, code):
        super().__init__(code, failure_kind="protocol")


class SSEParser:
    """One streamed choice, including fragmented indexed function calls."""

    def __init__(self):
        self.pending = ""
        self.event_data = []
        self.content = []
        self.reasoning = []
        self.calls = {}
        self.model = None
        self.usage = None
        self.finish_reason = None
        self.done = False
        self.events = 0

    def feed(self, text):
        self.pending += text
        while "\n" in self.pending and not self.done:
            line, self.pending = self.pending.split("\n", 1)
            line = line.removesuffix("\r")
            if not line and self.event_data:
                self.accept("\n".join(self.event_data))
                self.event_data = []
            elif line.startswith("data:"):
                self.event_data.append(line[5:].removeprefix(" "))

    def accept(self, text):
        if text == "[DONE]":
            self.done = True
            return
        try:
            event = loads(text)
        except EvalError:
            raise ProtocolError("invalid_sse_json") from None
        self.events += 1
        if not isinstance(event, dict):
            raise ProtocolError("invalid_sse_event")
        if event.get("error"):
            raise TransportError("provider_sse_error")
        model = event.get("model")
        if model is not None:
            if not isinstance(model, str) or self.model is not None and self.model != model:
                raise ProtocolError("inconsistent_stream_model")
            self.model = model
        if event.get("usage") is not None:
            self.usage = event["usage"]
        choices = event.get("choices", [])
        if not isinstance(choices, list) or len(choices) > 1:
            raise ProtocolError("unexpected_stream_choices")
        if not choices:
            return
        choice = choices[0]
        if not isinstance(choice, dict) or choice.get("index", 0) != 0:
            raise ProtocolError("unexpected_stream_choice")
        finish = choice.get("finish_reason")
        if finish is not None:
            if self.finish_reason is not None and self.finish_reason != finish:
                raise ProtocolError("inconsistent_finish_reason")
            self.finish_reason = finish
        delta = choice.get("delta", {})
        if not isinstance(delta, dict):
            raise ProtocolError("invalid_stream_delta")
        for field, target in (("content", self.content), ("reasoning_content", self.reasoning)):
            part = delta.get(field)
            if part is not None:
                if not isinstance(part, str):
                    raise ProtocolError("invalid_stream_text")
                target.append(part)
        fragments = delta.get("tool_calls") or []
        if not isinstance(fragments, list):
            raise ProtocolError("invalid_tool_fragments")
        for fragment in fragments:
            if not isinstance(fragment, dict) or type(fragment.get("index")) is not int or not 0 <= fragment["index"] < 8:
                raise ProtocolError("invalid_tool_index")
            call = self.calls.setdefault(fragment["index"], {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
            if fragment.get("type") not in (None, "function"):
                raise ProtocolError("invalid_tool_type")
            if fragment.get("id") is not None:
                if not isinstance(fragment["id"], str):
                    raise ProtocolError("invalid_tool_id")
                call["id"] += fragment["id"]
            function = fragment.get("function", {})
            if not isinstance(function, dict):
                raise ProtocolError("invalid_tool_function")
            for field in ("name", "arguments"):
                if function.get(field) is not None:
                    if not isinstance(function[field], str):
                        raise ProtocolError("invalid_tool_text")
                    call["function"][field] += function[field]

    def finish(self):
        if not self.done:
            raise TransportError("truncated_sse", retryable=True)
        if not self.model or self.finish_reason is None:
            raise ProtocolError("missing_stream_metadata")
        message = {"role": "assistant", "content": "".join(self.content),
                   "reasoning_content": "".join(self.reasoning)}
        if self.calls:
            if set(self.calls) != set(range(len(self.calls))):
                raise ProtocolError("noncontiguous_tool_indexes")
            calls = [self.calls[i] for i in sorted(self.calls)]
            ids = [c["id"] for c in calls]
            if any(not i for i in ids) or len(set(ids)) != len(ids) or any(not c["function"]["name"] for c in calls):
                raise ProtocolError("invalid_complete_tools")
            message["tool_calls"] = calls
        if bool(self.calls) != (self.finish_reason == "tool_calls"):
            raise ProtocolError("tool_finish_mismatch")
        return {"model": self.model, "usage": self.usage, "finish_reason": self.finish_reason,
                "message": message, "representation": "assembled_from_sse"}


def stream_transport(payload, key, output, *, timeout_seconds, idle_seconds):
    """Only official endpoint; no redirects; no credential-bearing exceptions."""
    output = Path(output)
    request = urllib.request.Request(ENDPOINT, data=dumps(payload).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json", "Accept": "text/event-stream", "Authorization": "Bearer " + key})
    start = time.monotonic()
    deadline = None if timeout_seconds is None else start + timeout_seconds
    parser = SSEParser()
    decoder = codecs.getincrementaldecoder("utf-8")()
    raw = bytearray()
    response = None
    metadata = {"http_status": None, "error": None, "retryable": False, "failure_kind": None, "first_byte_seconds": None}
    result = None
    try:
        connect_timeout = idle_seconds if timeout_seconds is None else min(idle_seconds, timeout_seconds)
        response = urllib.request.build_opener(NoRedirect()).open(request, timeout=connect_timeout)
        metadata["http_status"] = response.status
        if response.status != 200:
            raise TransportError("unexpected_http_status")
        if "text/event-stream" not in response.getheader("Content-Type", "").lower():
            raise ProtocolError("unexpected_content_type")
        active_socket = getattr(getattr(response.fp, "raw", None), "_sock", None)
        if active_socket is None:
            raise TransportError("stream_socket_unavailable")
        last_progress = start
        while not parser.done:
            remaining = idle_seconds if deadline is None else deadline - time.monotonic()
            if remaining <= 0:
                raise TransportError("total_stream_timeout", retryable=True)
            active_socket.settimeout(min(idle_seconds, remaining))
            chunk = response.read1(8192)
            if not chunk:
                break
            if metadata["first_byte_seconds"] is None:
                metadata["first_byte_seconds"] = round(time.monotonic() - start, 6)
            raw.extend(chunk)
            if len(raw) > MAX_STREAM_BYTES:
                raise TransportError("stream_size_limit")
            parser.feed(decoder.decode(chunk))
            if time.monotonic() - last_progress >= 30:
                print(dumps({"event": "stream_progress", "bytes": len(raw), "seconds": round(time.monotonic() - start, 2)}), flush=True)
                last_progress = time.monotonic()
        # EOF must not complete a partial event or turn its unfinished UTF-8 into
        # a protocol error. Only a fully delimited [DONE] completes the stream.
        if parser.done:
            decoder.decode(b"", final=True)
        result = parser.finish()
    except urllib.error.HTTPError as error:
        metadata.update(http_status=error.code, error="HTTP_" + str(error.code), retryable=error.code in (408, 429, 500, 502, 503, 504), failure_kind="transport")
    except TransportError as error:
        metadata.update(error=str(error), retryable=error.retryable, failure_kind=error.failure_kind)
    except http.client.IncompleteRead as error:
        # Partial bytes are audit evidence only, even if they contain [DONE].
        raw.extend(error.partial[:max(0, MAX_STREAM_BYTES + 1 - len(raw))])
        if metadata["first_byte_seconds"] is None and raw:
            metadata["first_byte_seconds"] = round(time.monotonic() - start, 6)
        exceeded = len(raw) > MAX_STREAM_BYTES
        metadata.update(error="stream_size_limit" if exceeded else "incomplete_http_read",
                        retryable=not exceeded, failure_kind="transport")
    except (TimeoutError, socket.timeout, urllib.error.URLError, ConnectionError):
        metadata.update(error="network_timeout_or_connection_failure", retryable=True, failure_kind="transport")
    except UnicodeError:
        metadata.update(error="invalid_stream_encoding", failure_kind="protocol")
    except Exception as error:
        metadata.update(error=type(error).__name__, failure_kind="transport")
    finally:
        if response is not None:
            response.close()
        metadata.update(duration_seconds=round(time.monotonic() - start, 6), received_bytes=len(raw),
                        events=parser.events, done_received=parser.done, finish_reason=parser.finish_reason,
                        returned_model=parser.model, usage=parser.usage, representation="raw_sse")
        safe_raw = redact(raw.decode("utf-8", errors="replace"), [key])
        with (output / "raw.sse.txt").open("x", encoding="utf-8", newline="") as handle:
            handle.write(safe_raw)
        metadata["raw_sse_sha256"] = sha256(safe_raw.encode("utf-8"))
        save(output / "stream.json", metadata, [key])
    if result is None:
        raise TransportError(metadata["error"] or "interrupted_stream", retryable=metadata["retryable"], metadata=metadata,
                             failure_kind=metadata["failure_kind"] or "transport")
    result["transport"] = metadata
    return result


def make_payload(data, prompt, arm, catalog, messages=None):
    if arm not in ARM_INSTRUCTIONS:
        raise EvalError("unknown arm")
    if messages is None:
        supplied = data if arm == "A" else {**data, "catalog": {"grade_mappings": data["catalog"]["grade_mappings"], "index": catalog.index_payload()}}
        messages = [{"role": "system", "content": prompt + "\n\n" + ARM_INSTRUCTIONS[arm]},
                    {"role": "user", "content": dumps(supplied)}]
    payload = {**MODEL_PARAMETERS, "messages": messages}
    if arm == "B":
        from src.agent_catalog import QUERY_BATCH_SCHEMA
        payload["tools"] = [{"type": "function", "function": {"name": "query_batch",
            "description": "只读批量目录检索。空条件仅在本次提供的目录范围内扩大检索；全部候选或显式分页；未知查询保留，不构成业务确认。",
            "parameters": QUERY_BATCH_SCHEMA}}]
        payload["tool_choice"] = "auto"
    return payload


def check_answer(text, data, catalog, allowed_ids):
    """Keep original output untouched; validity never establishes semantic truth."""
    parsed = None
    structure, business = [], []
    try:
        parsed = loads(text)
        structure = validate_answer(parsed, data)
    except (EvalError, TypeError, KeyError, IndexError, AttributeError):
        structure = ["invalid_answer_json_or_shape"]
    validation = None
    if not structure:
        validation = catalog.validate_proposal(parsed["items"], allowed_catalog_ids=allowed_ids)
        # Catalog module owns deterministic identity checks, workflow owns evidence.
        business = validation.get("errors", validation.get("issues", []))
        if not validation.get("proposal_valid", validation.get("valid", False)) and not business:
            business = ["catalog_consistency_failed"]
    unresolved = []
    if not structure:
        unresolved.extend(parsed["unresolved"])
        unresolved.extend(v for item in parsed["items"] for v in item["unresolved"])
        unresolved.extend(row["reason"] for row in parsed["coverage"] if row["disposition"] == "unresolved")
        for row in (validation or {}).get("items", []):
            unresolved.extend(issue["message"] for issue in row.get("unresolved", []))
    status = "structure_failed" if structure else "business_failed" if business else "unresolved" if unresolved else "proposal_valid"
    return {"status": status, "parsed": parsed, "structure_errors": structure, "business_errors": business,
            "catalog_validation": validation, "unresolved": unresolved, "quote_authorized": False,
            "semantic_accuracy_verified": False}


def validate_config(config, *, allow_unbounded=False):
    keys = {"max_requests", "max_calls_per_sample", "total_seconds", "sample_seconds", "request_seconds", "idle_seconds",
            "input_per_million", "output_per_million", "currency", "budget", "expected_manifest_sha256"}
    if not isinstance(config, dict) or set(config) - keys:
        raise EvalError("unexpected configuration field")
    for name, ceiling in (("max_requests", 72), ("max_calls_per_sample", 4)):
        if allow_unbounded and name == "max_requests" and name in config and config[name] is None:
            continue
        if type(config.get(name)) is not int or not 1 <= config[name] <= ceiling:
            raise EvalError("invalid " + name)
    for name, ceiling in (("total_seconds", 86400), ("sample_seconds", 7200), ("request_seconds", 3600), ("idle_seconds", 600)):
        value = config.get(name)
        if allow_unbounded and name in {"total_seconds", "sample_seconds", "request_seconds"} and name in config and value is None:
            continue
        if type(value) not in (int, float) or not 0 < value <= ceiling:
            raise EvalError("invalid " + name)
    if config["request_seconds"] is not None and config["idle_seconds"] > config["request_seconds"]:
        raise EvalError("idle timeout exceeds request timeout")
    if config.get("currency") != "CNY":
        raise EvalError("explicit CNY rates required")
    for name in ("input_per_million", "output_per_million", "budget"):
        if name == "budget" and config.get(name) is None:
            continue
        try:
            number = Decimal(str(config[name]))
            if not number.is_finite() or number <= 0:
                raise ValueError
        except (ValueError, ArithmeticError, KeyError):
            raise EvalError("invalid " + name) from None


def _event(ledger, value, key):
    ledger.write(redact(dumps(value), [key]) + "\n")
    ledger.flush()
    os.fsync(ledger.fileno())


def _usage_total(rounds):
    totals = {k: 0 for k in ("prompt_tokens", "completion_tokens", "total_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens")}
    missing = 0
    for row in rounds:
        usage = row.get("usage")
        if not isinstance(usage, dict) or any(type(usage.get(k)) is not int or usage[k] < 0 for k in ("prompt_tokens", "completion_tokens")):
            missing += 1
            continue
        for field in totals:
            if type(usage.get(field)) is int and usage[field] >= 0:
                totals[field] += usage[field]
        if "total_tokens" not in usage:
            totals["total_tokens"] += usage["prompt_tokens"] + usage["completion_tokens"]
    return {"known_usage_totals": totals, "requests_without_usage": missing,
            "usage_complete": missing == 0, "billing_note": "Usage is provider-reported; reservations and rate estimates are not an actual bill. Missing usage has unknown cost."}


def _query_fingerprint(arguments):
    # Model-generated query IDs and filter ordering do not add search information.
    queries = arguments.get("queries")
    if not isinstance(queries, list):
        raise TransportError("invalid_tool_queries")
    canonical = []
    for query in queries:
        if not isinstance(query, dict):
            raise TransportError("invalid_tool_query")
        canonical.append({k: sorted(v) if isinstance(v, list) and all(isinstance(x, str) for x in v) else v
                          for k, v in sorted(query.items()) if k != "query_id"})
    return sha256(sorted(canonical, key=dumps))


def _run_sample(data, prompt, arm, catalog, directory, *, config, key, transport, state, ledger, deadline, master_snapshot, on_stage=None):
    if arm == "B":
        available_requests = config["max_calls_per_sample"]
        if config["max_requests"] is not None:
            available_requests = min(available_requests, config["max_requests"] - state["requests"])
        prompt += (f"\n本来源最多可用 {available_requests} 次模型请求（含工具后续回合和网络重试，本地工具执行本身不计数）；最后一次须提交最终 JSON，"
                   "证据不足时保留未决，不得猜测业务事实。")
    messages = make_payload(data, prompt, arm, catalog)["messages"]
    allowed_ids = {p["catalog_id"] for p in data["catalog"]["products"]} if arm == "A" else set()
    seen_queries = set()
    seen_tool_ids = set()
    rounds = []
    validation = None
    status = "call_limit"
    started = time.monotonic()
    network_retry_used = False
    retry_stop_reason = None
    for number in range(1, config["max_calls_per_sample"] + 1):
        if master_snapshot is not None:
            try:
                master_snapshot.assert_current()
            except (ValueError, OSError):
                status = "stale_source"
                break
        sample_remaining = (float("inf") if config["sample_seconds"] is None
                            else config["sample_seconds"] - (time.monotonic() - started))
        if sample_remaining <= 0:
            status = "sample_timeout"
            break
        remaining = min(deadline - time.time(), sample_remaining)
        if remaining <= 0 or (config["max_requests"] is not None and state["requests"] >= config["max_requests"]):
            status = "batch_limit"
            break
        payload = make_payload(data, prompt, arm, catalog, messages)
        encoded = dumps(payload).encode("utf-8")
        if len(encoded) > MAX_REQUEST_BYTES:
            status = "request_size_limit"
            break
        input_bound = len(encoded) + 4096
        reserve = (Decimal(input_bound) * Decimal(config["input_per_million"]) + Decimal(65536 + 64) * Decimal(config["output_per_million"])) / Decimal(1000000)
        if config.get("budget") is not None and state["reserved"] + reserve > Decimal(config["budget"]):
            status = "budget_limit"
            break
        attempt = directory / ("round-%02d" % number)
        attempt.mkdir()
        save(attempt / "payload.json", payload, [key])
        state["requests"] += 1
        state["reserved"] += reserve
        reserved = {"event": "reserve", "ordinal": state["requests"], "sample_arm": directory.name,
                    "round": number, "request_sha256": sha256(encoded), "input_token_bound": input_bound,
                    "amount": str(reserve), "total_reserved": str(state["reserved"]), "currency": "CNY",
                    "started_epoch": time.time()}
        save(attempt / "request.json", reserved)
        _event(ledger, reserved, key)
        began = time.monotonic()
        row = {"round": number, "request_sha256": reserved["request_sha256"], "usage": None,
               "status": "request_failed", "returned_model": None, "finish_reason": None,
               "network_retry": status == "retry_pending",
               "master_before_sha256": master_snapshot.master_fingerprint if master_snapshot else None,
               "master_after_sha256": None}
        stage = "transport"
        try:
            if row["network_retry"]:
                network_retry_used = True
            if on_stage:
                on_stage(f"模型网络请求失败，正在重试（本来源请求 {number}/{config['max_calls_per_sample']}）…"
                         if row["network_retry"] else "模型正在推理…")
            timeout_seconds = min((value for value in (config["request_seconds"], remaining)
                                   if value is not None and value != float("inf")), default=None)
            response = transport(payload, key, attempt, timeout_seconds=timeout_seconds, idle_seconds=config["idle_seconds"])
            # Transport injection receives the same exact payload as real execution.
            if not isinstance(response, dict) or len(dumps(response).encode("utf-8")) > MAX_STREAM_BYTES:
                raise TransportError("invalid_transport_response")
            save(attempt / "response.json", response, [key])
            row.update(usage=response.get("usage"), returned_model=response.get("model"), finish_reason=response.get("finish_reason"), status="response_received")
            stage = "protocol"
            if not isinstance(response.get("model"), str) or not response["model"]:
                raise TransportError("missing_returned_model")
            message = response["message"]
            if not isinstance(message, dict) or message.get("role") != "assistant" or not isinstance(message.get("content"), str) or not isinstance(message.get("reasoning_content"), str):
                raise TransportError("invalid_assistant_message")
            row.update(answer_characters=len(message["content"]), reasoning_characters=len(message["reasoning_content"]))
            calls = message.get("tool_calls") or []
            if response["finish_reason"] == "tool_calls":
                if arm != "B" or not isinstance(calls, list) or not 1 <= len(calls) <= 8:
                    raise TransportError("unexpected_tool_calls")
                # Full assistant message replay is required by DeepSeek thinking tools.
                messages.append(message)
                tool_results = []
                new_information = False
                for call in calls:
                    if not isinstance(call, dict) or set(call) != {"id", "type", "function"} or call["type"] != "function" or not isinstance(call["id"], str) or not call["id"] or call["id"] in seen_tool_ids:
                        raise TransportError("invalid_or_duplicate_tool_call")
                    seen_tool_ids.add(call["id"])
                    function = call["function"]
                    if not isinstance(function, dict) or set(function) != {"name", "arguments"} or function["name"] != "query_batch":
                        raise TransportError("unrecognized_tool")
                    args = loads(function["arguments"])
                    if not isinstance(args, dict) or set(args) != {"queries"}:
                        raise TransportError("invalid_tool_arguments")
                    query_hash = _query_fingerprint(args)
                    if query_hash in seen_queries:
                        tool_result = {"error": "repeated_query_no_new_information"}
                    else:
                        seen_queries.add(query_hash)
                        # Catalog owns parameter bounds and result completeness.
                        stage = "tool"
                        if on_stage:
                            on_stage("正在核对产品目录…")
                        tool_result = catalog.query_batch(args["queries"])
                        stage = "protocol"
                        new_information = True
                        for result in tool_result["results"]:
                            allowed_ids.update(p["catalog_id"] for p in result["candidates"])
                    tool_results.append({"tool_call_id": call["id"], "arguments": args, "result": tool_result})
                    messages.append({"role": "tool", "tool_call_id": call["id"], "content": dumps(tool_result)})
                save(attempt / "tool-results.json", tool_results, [key])
                row["status"] = "tools_completed"
                status = "call_limit" if new_information else "no_new_information"
            else:
                if calls:
                    raise TransportError("tool_finish_mismatch")
                validation = check_answer(message["content"], data, catalog, allowed_ids)
                if response["finish_reason"] != "stop":
                    validation["structure_errors"].append("non_stop_finish")
                    validation["status"] = "structure_failed"
                save(attempt / "validation.json", validation, [key])
                status = row["status"] = validation["status"]
        except TransportError as error:
            if error.failure_kind == "protocol":
                stage = "protocol"
            row.update(error=str(error), retryable=error.retryable, failure_kind=error.failure_kind)
            if error.metadata:
                row.update(usage=error.metadata.get("usage"), returned_model=error.metadata.get("returned_model"), finish_reason=error.metadata.get("finish_reason"))
            # At most one network retry for this entire sample/arm; all count.
            batch_exhausted = config["max_requests"] is not None and state["requests"] >= config["max_requests"]
            can_retry = (stage == "transport" and error.retryable and not network_retry_used
                         and number < config["max_calls_per_sample"] and not batch_exhausted)
            status = "retry_pending" if can_retry else "request_failed" if stage == "transport" else stage + "_failed"
            row["status"] = "request_failed" if stage == "transport" else stage + "_failed"
            if stage == "transport" and error.retryable and not can_retry:
                retry_stop_reason = ("batch_request_limit" if batch_exhausted
                                     else "source_request_limit" if number >= config["max_calls_per_sample"]
                                     else "network_retry_limit")
        except Exception as error:
            row.update(error=type(error).__name__, retryable=False)
            status = row["status"] = stage + "_failed" if stage in {"protocol", "tool"} else "workflow_failed"
        if master_snapshot is not None:
            try:
                master_snapshot.assert_current()
                row["master_after_sha256"] = master_snapshot.master_fingerprint
            except (ValueError, OSError):
                row["status_before_source_check"] = row["status"]
                status = row["status"] = "stale_source"
        row["duration_seconds"] = round(time.monotonic() - began, 6)
        save(attempt / "result.json", row, [key])
        rounds.append(row)
        _event(ledger, {"event": "result", "ordinal": state["requests"], "sample_arm": directory.name, **row}, key)
        if status == "retry_pending":
            continue
        if row["status"] != "tools_completed" or status == "no_new_information":
            break
    summary = {"input_id": directory.name.rsplit("__", 1)[0], "input_sha256": sha256(data), "arm": arm, "status": status, "rounds": rounds,
               "duration_seconds": round(time.monotonic() - started, 6), "model_requests": len(rounds),
               "allowed_catalog_ids": sorted(allowed_ids), "validation": validation,
               "quote_authorized": False, **_usage_total(rounds)}
    if rounds and rounds[-1].get("retryable") and status in {"batch_limit", "budget_limit", "sample_timeout", "stale_source", "request_size_limit"}:
        retry_stop_reason = status
    if retry_stop_reason:
        summary["retry_stop_reason"] = retry_stop_reason
    save(directory / "summary.json", summary, [key])
    return summary


def run_inquiry(documents, prompt, output, *, config, master_snapshot,
                transport=None, environ=None, started_epoch=None, on_stage=None):
    """Per-source requests; batch accounting and explicitly configured gates persist."""
    from .agent_catalog import inquiry_categories

    validate_config(config, allow_unbounded=True)
    master_snapshot.assert_current()
    if not isinstance(documents, list) or not documents:
        raise EvalError("empty inquiry sources")
    entries, source_ids = [], set()
    document_fields = {"source_id", "source_name", "source_type", "images", "units"}
    unit_fields = {"unit_id", "text", "page", "ocr_page", "bbox", "block_label", "block_order", "line", "sheet", "row", "reading_method"}
    for document in documents:
        if not isinstance(document, dict) or set(document) != document_fields or document["images"] != []:
            raise EvalError("invalid or image-bearing inquiry source")
        ident = document["source_id"]
        if not isinstance(ident, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", ident) or ident in source_ids:
            raise EvalError("invalid or duplicate source id")
        source_ids.add(ident)
        if any(not isinstance(document[field], str) or not document[field].strip() for field in ("source_name", "source_type")):
            raise EvalError("invalid source metadata")
        units = document["units"]
        if not isinstance(units, list) or not units:
            raise EvalError("empty source evidence")
        unit_ids = set()
        for unit in units:
            if not isinstance(unit, dict) or set(unit) - unit_fields or not {"unit_id", "text"} <= unit.keys():
                raise EvalError("unexpected evidence fields")
            if any(not isinstance(unit[field], str) or not unit[field].strip() for field in ("unit_id", "text")) or unit["unit_id"] in unit_ids:
                raise EvalError("invalid or duplicate evidence unit")
            unit_ids.add(unit["unit_id"])
        source_catalog = master_snapshot.restrict_categories(inquiry_categories(document))
        data = {"schema_version": "crp-inquiry-v1", "document": document, "catalog": source_catalog.to_payload()}
        reject_private(data)
        # Validate JSON serializability before any provider call or evidence write.
        dumps(data)
        entries.append((data, source_catalog))
    environ = os.environ if environ is None else environ
    key = environ.get("DEEPSEEK_API_KEY", "")
    if not isinstance(key, str) or not key.strip() or "\r" in key or "\n" in key:
        raise EvalError("required DEEPSEEK_API_KEY credential unavailable")
    transport = stream_transport if transport is None else transport
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started_epoch = time.time() if started_epoch is None else started_epoch
    deadline = float("inf") if config["total_seconds"] is None else started_epoch + config["total_seconds"]
    save(output / "run.json", {"schema_version": "crp-inquiry-run-v1", "created_epoch": started_epoch,
         "model_parameters": MODEL_PARAMETERS, "config": config, "endpoint": ENDPOINT,
         "master_fingerprint": master_snapshot.master_fingerprint, "prompt_sha256": sha256(prompt),
         "sources": documents}, [key])
    state = {"requests": 0, "reserved": Decimal(0)}
    summaries, stop_reason = [], None
    with (output / "ledger.jsonl").open("x", encoding="utf-8") as ledger:
        for data, source_catalog in entries:
            ident = data["document"]["source_id"]
            directory = output / ident
            directory.mkdir()
            save(directory / "input.json", data, [key])
            if (stop_reason or time.time() >= deadline
                    or (config["max_requests"] is not None and state["requests"] >= config["max_requests"])):
                summary = {"input_id": ident, "status": "not_processed", "reason": stop_reason or "batch_limit",
                           "validation": None, "model_requests": 0, "quote_authorized": False}
                save(directory / "summary.json", summary, [key])
            else:
                summary = _run_sample(data, prompt, "B", source_catalog, directory, config=config,
                                      key=key, transport=transport, state=state, ledger=ledger,
                                      deadline=deadline, master_snapshot=master_snapshot, on_stage=on_stage)
                source_exhausted = summary["status"] == "call_limit" or (
                    summary["status"] == "request_failed"
                    and summary.get("retry_stop_reason") in {"source_request_limit", "network_retry_limit"}
                    and summary["rounds"][-1].get("failure_kind") == "transport"
                    and summary["rounds"][-1].get("retryable") is True
                )
                if summary["status"] not in {"proposal_valid", "unresolved"} and not source_exhausted:
                    stop_reason = summary["status"]
            summaries.append(summary)
    try:
        master_snapshot.assert_current()
    except (ValueError, OSError):
        for summary in summaries:
            summary["status_before_final_source_check"] = summary["status"]
            summary["status"] = "stale_source"
    report = {"samples": summaries, "model_requests": state["requests"],
              "reserved_cny": str(state["reserved"]), "master_fingerprint": master_snapshot.master_fingerprint,
              "evidence_directory": str(output), "quote_authorized": False}
    save(output / "summary.json", report, [key])
    return loads(redact(dumps(report), [key]))


def run_ab(config, entries, prompt, output, *, subset=None, execute=False, resume=False,
           transport=stream_transport, environ=None, source_hashes=None, master_snapshot=None):
    """Serial, alternating A/B. Resume skips terminal evidence; incomplete arms block.

    No answer repair is performed: the first final JSON is the scored raw answer.
    Extra calls are only tool continuation or one network retry, never success fishing.
    """
    from src.agent_catalog import CatalogSnapshot
    validate_config(config)
    environ = os.environ if environ is None else environ
    if not entries:
        raise EvalError("empty inputs")
    ids = [entry[0] for entry in entries]
    if any(not isinstance(i, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", i) for i in ids) or len(ids) != len(set(ids)):
        raise EvalError("invalid input identifiers")
    selected = set(ids if subset is None else subset)
    if not selected or not selected <= set(ids):
        raise EvalError("unknown or empty subset")
    catalog = CatalogSnapshot.from_eval_catalog(entries[0][2]["catalog"])
    for _, _, data in entries:
        reject_private(data)
        if data["document"].get("images") != [] or data["catalog"] != entries[0][2]["catalog"]:
            raise EvalError("image payload or catalog snapshot mismatch")
    schedule = [(ident, arm) for index, ident in enumerate(ids) if ident in selected
                for arm in (("A", "B") if index % 2 == 0 else ("B", "A"))]
    identity = {"schema_version": "crp-agent-catalog-ab-v1", "inputs": [{"id": i, "file_sha256": h, "content_sha256": sha256(d)} for i, h, d in entries],
                "business_prompt_sha256": sha256(prompt.encode("utf-8")), "arm_instructions_sha256": sha256(ARM_INSTRUCTIONS),
                "model_parameters": MODEL_PARAMETERS, "model_parameters_sha256": sha256(MODEL_PARAMETERS),
                "config": config, "config_sha256": sha256(config), "source_hashes": source_hashes or {},
                "master_workbook_sha256": master_snapshot.master_fingerprint if master_snapshot else None,
                "catalog_fingerprint": catalog.catalog_fingerprint, "endpoint": ENDPOINT,
                "format_policy": "plain official tools; no response_format; final JSON strictly validated without cleaning"}
    plan = {"identity": identity, "schedule": schedule, "minimum_model_requests": len(schedule),
            "maximum_model_requests": min(len(schedule) * config["max_calls_per_sample"], config["max_requests"]),
            "mode": "execute" if execute else "dry_run", "resume": resume,
            "billing_note": "Conservative 2/8 CNY rates can be used for reservation; cache usage is retained. No recharge or balance bypass. HTTP 402 stops immediately."}
    if not execute:
        return plan
    if transport is stream_transport and master_snapshot is None:
        raise EvalError("real execution requires the verified master snapshot")
    key = environ.get("DEEPSEEK_API_KEY", "")
    if not isinstance(key, str) or not key.strip() or "\r" in key or "\n" in key:
        raise EvalError("required DEEPSEEK_API_KEY credential unavailable")
    output = Path(output)
    if resume:
        original = loads((output / "run.json").read_text(encoding="utf-8"))
        if original["identity"] != identity:
            raise EvalError("resume fingerprint mismatch; create a separately labelled run")
    else:
        output.mkdir(parents=True, exist_ok=False)
        original = {"identity": identity, "created_epoch": time.time()}
        save(output / "run.json", original)
    deadline = original["created_epoch"] + config["total_seconds"]
    lock = output / "active.lock"
    # One writer. A crash leaves the lock as an explicit recovery boundary.
    with lock.open("x", encoding="utf-8") as handle:
        handle.write("single writer\n")
    try:
        ledger_path = output / "ledger.jsonl"
        previous = [loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()] if ledger_path.exists() else []
        if any(e.get("status") == "stale_source" for e in previous):
            raise EvalError("stale-source run is closed; cannot resume")
        reservations = [e for e in previous if e.get("event") == "reserve"]
        state = {"requests": len(reservations), "reserved": sum((Decimal(e["amount"]) for e in reservations), Decimal(0))}
        entries_by_id = {i: d for i, _, d in entries}
        summaries = []
        with ledger_path.open("a" if resume else "x", encoding="utf-8") as ledger:
            _event(ledger, {"event": "invocation", "schedule": schedule, "epoch": time.time(), "resume": resume}, key)
            for ident, arm in schedule:
                directory = output / (ident + "__" + arm)
                if directory.exists():
                    if not resume:
                        raise EvalError("sample evidence already exists")
                    if not (directory / "summary.json").is_file():
                        raise EvalError("incomplete sample evidence; unknown billed attempt must not be replayed")
                    summaries.append(loads((directory / "summary.json").read_text(encoding="utf-8")))
                    continue
                if state["requests"] >= config["max_requests"] or time.time() >= deadline:
                    break
                directory.mkdir()
                save(directory / "input.json", {"input_id": ident, "arm": arm, "input_sha256": sha256(entries_by_id[ident])})
                summary = _run_sample(entries_by_id[ident], prompt, arm, catalog, directory, config=config,
                                      key=key, transport=transport, state=state, ledger=ledger, deadline=deadline,
                                      master_snapshot=master_snapshot)
                summaries.append(summary)
                _event(ledger, {"event": "sample", "sample_arm": directory.name, "status": summary["status"]}, key)
                print(dumps({"event": "sample_completed", "input_id": ident, "arm": arm, "status": summary["status"], "requests": summary["model_requests"]}), flush=True)
                if summary["status"] in {"request_failed", "workflow_failed", "budget_limit", "batch_limit", "stale_source"}:
                    break
            all_rounds = [loads(p.read_text(encoding="utf-8")) for p in output.glob("*__*/round-*/result.json")]
            report = {"identity_sha256": sha256(identity), "selected_schedule": schedule, "samples": summaries,
                      "all_reserved_requests": state["requests"], "reservation_total_cny": str(state["reserved"]),
                      "all_recorded_request_seconds": sum(r["duration_seconds"] for r in all_rounds), **_usage_total(all_rounds)}
            report_path = output / ("summary-%03d.json" % sum(e.get("event") == "invocation" for e in previous))
            save(report_path, report, [key])
            return report
    finally:
        lock.unlink()
