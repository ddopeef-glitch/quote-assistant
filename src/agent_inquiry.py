"""Raw local evidence → bounded catalog Agent → unconfirmed UI proposals."""
from __future__ import annotations

import os
from pathlib import Path
import time
from uuid import uuid4

from .agent_catalog import CatalogSnapshot
from .agent_context import inquiry_contexts, item_context
from . import agent_workflow as workflow
from .core import ProductRecord, normalize_price_marker
from .input_router import MultiFrameImageError, extract_evidence_units
from .usage_diagnostics import emit_diagnostic


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "data" / "agent_runs"
PROMPT_PATH = ROOT / "benchmarks" / "prompts" / "inquiry_semantic_v2.txt"
# Historical accounting estimates only; not current provider prices or a bill.
RUN_CONFIG = {"max_requests": None, "max_calls_per_sample": 2, "total_seconds": None,
              "sample_seconds": None, "request_seconds": None, "idle_seconds": 60,
              "input_per_million": "2", "output_per_million": "8", "currency": "CNY", "budget": None}
REVIEW_WARNING = "模型提案未经人工确认；请对照原始证据核对分类、产品、规格、箱数与档位"
INPUT_NOTE = "\n当前输入说明：同一 source_id、page 上 reading_method 为 pdf_text 与 local_ocr 的内容是同页互补证据，可能重复描述同一产品；不得重复计成两份询价。pdf_text 的 bbox 使用 PDF 坐标，local_ocr 的 bbox 使用渲染并规范化后的图像坐标，不能跨 reading_method 直接比较 bbox 数值；请分别理解各自布局，按同页内容合并核对，冲突保留 unresolved。"


class InquiryError(ValueError):
    """Safe user-facing failure; no provider exception or credential contents."""


def _pending(document, units, reason):
    return ProductRecord(
        source_type=document["source_type"], source_name=document["source_name"],
        source_location="、".join(unit["unit_id"] for unit in units),
        original_text="\n".join(unit["text"] for unit in units),
        extraction_method="待人工处理", warnings=[reason, "此行原始字段与等级尚未核对，不能仅填写五列解除阻断；请重新识别，或明确删除此占位后继续"],
    )


def _records_from_report(documents, report, grade_markers):
    records, notices = [], []
    for document, sample in zip(documents, report["samples"], strict=True):
        name, status = document["source_name"], sample["status"]
        validation = sample.get("validation")
        if (status not in {"proposal_valid", "unresolved"} or not validation
                or validation["structure_errors"] or validation["business_errors"]):
            last_round = (sample.get("rounds") or [{}])[-1]
            failure = {"incomplete_http_read": "响应未完整接收", "truncated_sse": "响应未完整接收",
                       "network_timeout_or_connection_failure": "网络超时或连接中断"}.get(last_round.get("error"))
            stopped = {"batch_request_limit": "本次识别的模型请求次数已用尽，未再重试",
                       "source_request_limit": "本来源的模型请求次数已用尽，未再重试",
                       "network_retry_limit": "本来源的一次网络重试已用完，未再重试",
                       "batch_limit": "本次模型请求或时间限额已用尽，未再重试",
                       "sample_timeout": "本来源的显式时间限额已用尽，未再重试",
                       "budget_limit": "已达到显式费用预留上限，未再重试",
                       "stale_source": "主数据已变化，未再重试",
                       "request_size_limit": "请求内容超出大小限制，未再重试"}.get(sample.get("retry_stop_reason"))
            if status == "call_limit":
                stopped = "本来源的模型请求次数已用尽，未取得最终提案"
            detail = "".join("；" + value for value in (failure, stopped) if value)
            reason = f"{name}：模型未完成有效识别（{status}）{detail}，此来源待人工处理"
            notices.append(reason)
            records.append(_pending(document, document["units"], reason))
            continue
        parsed = validation["parsed"]
        if parsed["document_type"] == "price_matrix":
            reason = f"{name}：被识别为价格矩阵，未将其中数字作为询价箱数"
            notices.append(reason)
            records.append(_pending(document, document["units"], reason))
            continue
        units = {unit["unit_id"]: unit for unit in document["units"]}
        contexts = inquiry_contexts(document)
        checks = {item["item_id"]: item for item in validation["catalog_validation"]["items"]}
        notices.extend(f"{name}：{message}" for message in validation["unresolved"])
        if parsed["document_type"] in {"mixed", "unknown"}:
            notices.append(f"{name}：文档类型为 {parsed['document_type']}，请核对是否为完整询价")
        for item in parsed["items"]:
            check = checks[item["item_id"]]
            cited = list(dict.fromkeys(evidence["unit_id"] for evidence in item["evidence"]))
            warnings = [REVIEW_WARNING, *parsed["unresolved"], *item["unresolved"]]
            warnings.extend(issue["message"] for issue in check["unresolved"] + check["warnings"])
            warnings.extend(row["reason"] for row in parsed["coverage"]
                            if row["disposition"] == "unresolved" and item["item_id"] in row["item_ids"])
            category, title_grade, category_ambiguous, grade_ambiguous = item_context(item, contexts)
            item_unresolved = [*item["unresolved"], *(
                row["reason"] for row in parsed["coverage"]
                if row["disposition"] == "unresolved" and item["item_id"] in row["item_ids"])]
            # Free-form/document warnings are evidence, not per-item grade state.
            grade_unresolved = (any(issue["code"] == "unknown_grade" for issue in check["unresolved"])
                                or any(message.lstrip().startswith(("等级：", "等级:"))
                                       for message in item_unresolved))
            proposed_category = item["category"]
            marker = check.get("resolved_price_marker", item["price_marker"])
            if category_ambiguous:
                proposed_category = None
                warnings.append("本项分类标题/行文范围未决，请人工选择分类")
            else:
                if proposed_category is None:
                    proposed_category = category
                elif proposed_category != category:
                    proposed_category = None
                    warnings.append("分类提案与本项原文/标题范围冲突，请人工选择分类")
            grade_review = False
            if grade_unresolved or not item["grade_raw"] and grade_ambiguous:
                marker, grade_review = None, True
                warnings.append("本项等级或标题范围尚未核对，须人工选择等级，不能套用缺省 b")
            elif title_grade and marker != normalize_price_marker(title_grade, grade_markers):
                marker, grade_review = None, True
                warnings.append(f"本项有效等级标题为“{title_grade}”，等级提案缺失或冲突，请人工选择等级")
            elif not item["grade_raw"] and not title_grade:
                if marker in {"a", "A"}:
                    marker, grade_review = None, True
                    warnings.append("等级提案缺少原文或有效标题依据，请人工选择等级")
                else:
                    marker = "b"
            records.append(ProductRecord(
                source_type=document["source_type"], source_name=name, source_location="、".join(cited),
                original_text="\n".join(units[ident]["text"] for ident in cited),
                original_product_name=item["product_name_raw"], product_name=item["product_name"],
                original_specification=item["specification_raw"], specification=item["specification"],
                category=proposed_category, quantity_cartons=item["quantity_cartons"],
                original_grade_text=item["grade_raw"], price_marker=marker, grade_review_required=grade_review,
                extraction_method="DeepSeek 目录 Agent 提案", warnings=list(dict.fromkeys(warnings)),
            ))
        for row in parsed["coverage"]:
            if row["disposition"] == "unresolved" and not row["item_ids"]:
                records.append(_pending(document, [units[row["unit_id"]]], f"未决原文：{row['reason']}"))
        if not parsed["items"]:
            notices.append(f"{name}：未返回产品提案；全部原文和逐项覆盖说明保留在本次识别证据中")
    return records, list(dict.fromkeys(notices))


def recognize_inquiry(files, text, master_path, ocr, *, transport=None, environ=None,
                      output_root=None, output_directory=None, config=None, on_stage=None, on_diagnostic=None):
    """Each upload and the nonempty pasted text get a separate request allowance."""
    started = time.time()
    sources = list(files)
    if text.strip():
        sources.append(("粘贴文字.txt", text.encode("utf-8")))
    if not sources:
        emit_diagnostic(on_diagnostic, "recognition_rejected", stage="input", reason="empty_input")
        raise InquiryError("请先上传询价文件或粘贴询价文字")
    environ = os.environ if environ is None else environ
    key = environ.get("DEEPSEEK_API_KEY", "")
    if not isinstance(key, str) or not key.strip() or "\r" in key or "\n" in key:
        emit_diagnostic(on_diagnostic, "recognition_rejected", stage="credentials", reason="unavailable_credentials")
        raise InquiryError("当前进程未配置有效 DEEPSEEK_API_KEY 凭据，未调用模型；请配置后重新启动系统")
    documents, errors = [], []
    for index, (filename, data) in enumerate(sources, 1):
        name = Path(filename).name
        source = {"source_id": f"source-{index}", "source_name": name}
        emit_diagnostic(on_diagnostic, "source_started", **source)
        try:
            if on_stage:
                on_stage("正在读取询价文件…")
            units = extract_evidence_units(name, data, ocr, on_stage=on_stage,
                on_diagnostic=lambda event, details: emit_diagnostic(on_diagnostic, event, **source, **details))
        except MultiFrameImageError as error:
            emit_diagnostic(on_diagnostic, "source_failed", **source, stage="input_read", error_type=type(error).__name__, error=str(error))
            errors.append(f"{name}：检测到多帧图片，请拆分成静态图片后重传")
            continue
        except Exception as error:
            emit_diagnostic(on_diagnostic, "source_failed", **source, stage="input_read", error_type=type(error).__name__, error=str(error))
            # Only local input preparation runs here. Never echo raw third-party errors.
            detail = "；旧 .xls 请另存为 .xlsx" if name.lower().endswith(".xls") else ""
            errors.append(f"{name}（{type(error).__name__}{detail}）")
            continue
        documents.append({"source_id": f"source-{index}", "source_name": name,
                          "source_type": name.lower().rsplit(".", 1)[-1], "images": [], "units": units})
        emit_diagnostic(on_diagnostic, "source_read", document=documents[-1])
    if errors:
        emit_diagnostic(on_diagnostic, "recognition_rejected", stage="input_read", errors=errors, documents=documents)
        raise InquiryError("本次整体未发送模型，以下来源读取失败，需人工核对后重试：" + "、".join(errors))
    model_directory = Path(output_directory) if output_directory is not None else Path(output_root or RUNS_DIR) / uuid4().hex
    emit_diagnostic(on_diagnostic, "workflow_started", evidence_directory=str(model_directory), documents=documents)
    try:
        snapshot = CatalogSnapshot.from_master(master_path)
        report = workflow.run_inquiry(documents, PROMPT_PATH.read_text(encoding="utf-8") + INPUT_NOTE,
                                      model_directory,
                                      config=dict(RUN_CONFIG if config is None else config),
                                      master_snapshot=snapshot, transport=transport, environ=environ,
                                      started_epoch=started, on_stage=on_stage)
        snapshot.assert_current()
    except (ValueError, OSError) as error:
        emit_diagnostic(on_diagnostic, "workflow_failed", evidence_directory=str(model_directory), error_type=type(error).__name__, error=str(error))
        raise InquiryError(f"本次模型识别未能完成（{type(error).__name__}），未采用提案；请检查配置、主数据和识别证据后重试") from None
    if any(sample["status"] == "stale_source" for sample in report["samples"]):
        emit_diagnostic(on_diagnostic, "workflow_failed", stage="master_changed", report=report)
        raise InquiryError("识别期间主数据变化，本次提案已全部作废，请重新识别")
    records, notices = _records_from_report(documents, report, dict(snapshot.grade_mappings))
    emit_diagnostic(on_diagnostic, "workflow_returned", report=report, notices=notices)
    return records, {"documents": documents, "run": report, "notices": notices}
