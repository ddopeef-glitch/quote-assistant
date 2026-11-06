"""Fixed synthetic tool-call replay; no AI inference, API key or network."""
import json

from .agent_inquiry import recognize_inquiry
from .agent_workflow import dumps

DEMO_ROWS = (
    ("沟槽", "90°弯头", "108", "b", 2),
    ("玛钢", "三通", "25", "a", 3),
    ("球铁", "球铁三通", "25", "b", 1),
)
DEMO_TEXT = "\n".join(f"{category} {name} {spec} {marker} {quantity}箱"
                      for category, name, spec, marker, quantity in DEMO_ROWS)


def replay_transport(payload, key, output, **limits):
    """Replay only the exact bundled inquiry; never parse arbitrary user input."""
    document = json.loads(payload["messages"][1]["content"])["document"]
    if [unit["text"].strip() for unit in document["units"]] != DEMO_TEXT.splitlines():
        raise ValueError("离线回放仅支持内置合成询价")
    tools = [message for message in payload["messages"] if message["role"] == "tool"]
    message = {"role": "assistant", "content": "", "reasoning_content": "固定合成回放，不是模型推理"}
    if not tools:
        queries = [{"query_id": f"q{index}", "product_terms": [name], "specifications": [spec],
                    "categories": [category], "price_markers": [marker]}
                   for index, (category, name, spec, marker, _) in enumerate(DEMO_ROWS, 1)]
        message["tool_calls"] = [{"id": "demo-query", "type": "function", "function": {
            "name": "query_batch", "arguments": dumps({"queries": queries})}}]
    else:
        results = json.loads(tools[-1]["content"])["results"]
        items, coverage = [], []
        for index, (row, unit, result) in enumerate(zip(DEMO_ROWS, document["units"], results, strict=True), 1):
            category, name, spec, marker, quantity = row
            candidates = [candidate for candidate in result["candidates"] if candidate["product_name"] == name]
            if len(candidates) != 1:
                raise ValueError("合成目录已变化，无法回放预设唯一候选")
            ident = candidates[0]["catalog_id"]
            item_id = f"demo-{index}"
            items.append({"item_id": item_id, "source_id": document["source_id"],
                          "evidence": [{"unit_id": unit["unit_id"], "quote": unit["text"]}],
                          "product_name_raw": name, "specification_raw": spec,
                          "quantity_raw": f"{quantity}箱", "grade_raw": marker,
                          "category": category, "product_name": name, "specification": spec,
                          "quantity_cartons": quantity, "price_marker": marker,
                          "candidate_ids": [ident], "selected_candidate_id": ident, "unresolved": []})
            coverage.append({"source_id": document["source_id"], "unit_id": unit["unit_id"],
                             "disposition": "items", "item_ids": [item_id], "reason": "合成询价逐行回放"})
        message["content"] = dumps({"document_type": "inquiry", "items": items,
                                    "coverage": coverage, "unresolved": []})
    return {"model": "offline-synthetic-replay", "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            "finish_reason": "tool_calls" if not tools else "stop", "message": message}


def recognize_demo(master_path, *, output_root=None):
    records, audit = recognize_inquiry([], DEMO_TEXT, master_path, None,
                                      transport=replay_transport, environ={"DEEPSEEK_API_KEY": "offline-demo"},
                                      output_root=output_root)
    audit["demo"] = True
    audit["notices"].append("固定合成提案回放：未调用 AI；用于演示目录查询、校验、人工确认和出单。")
    return records, audit
