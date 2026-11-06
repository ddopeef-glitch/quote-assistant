from __future__ import annotations

from dataclasses import fields
from decimal import Decimal
from hashlib import sha256
from pathlib import Path

import pandas as pd
import streamlit as st

from config import APP_NAME, APP_VERSION, OUTPUT_DIR, ROOT
from src.core import (
    CUSTOMER_HISTORY_PATH,
    ProductRecord,
    confirm_inquiry,
    discover_master_data,
    generate_quotation,
    load_customer_names,
    load_master_data,
    master_data_fingerprint,
    match_price,
    positive_integer,
    render_quotation_preview,
    required_categories,
)
from src.agent_inquiry import InquiryError, recognize_inquiry
from src.ocr_pipeline import LocalOCRExtractor
from src.usage_diagnostics import Diagnostics
from src.demo import DEMO_TEXT, recognize_demo
from src.input_router import IMAGE_SUFFIXES, validate_image


st.set_page_config(page_title=f"{APP_NAME} {APP_VERSION}", page_icon="🧾", layout="wide")
st.title(APP_NAME)
st.caption("本地 OCR → 受限目录查询 → 人工确认 → 确定性计价 → PDF / PNG")
st.info("所有内置商品数值和单位名称均为虚构演示数据，不可作为实际报价依据。")

MASTER_DATA_DIR = ROOT / "主数据"
diagnostic_notice = st.empty()


def diagnose(event, **data):
    diagnostic = st.session_state.get("usage_diagnostics")
    if diagnostic:
        diagnostic.event(event, **data)
        show_diagnostic_warning()


def show_diagnostic_warning():
    diagnostic = st.session_state.get("usage_diagnostics")
    if diagnostic and diagnostic.errors:
        diagnostic_notice.warning("本次本地诊断记录不完整，部分信息未能保存。")


def snapshot(records):
    return [record.to_dict() for record in records]


def reset_task() -> None:
    reset_recognition_editor()
    st.session_state.pop("diagnostic_zip", None)
    for key in ("records", "recognized", "recognition_confirmed", "matched", "matched_fingerprint", "generated", "price_editor", "model_audit", "recognition_error", "recognition_status"):
        st.session_state.pop(key, None)


def reset_recognition_editor() -> None:
    st.session_state.recognition_editor_version = st.session_state.get("recognition_editor_version", 0) + 1


def clear_generated() -> None:
    st.session_state.pop("generated", None)


def records_from_frame(frame: pd.DataFrame, previous: list[ProductRecord], deleted_rows=()) -> list[ProductRecord]:
    result = []
    valid_fields = {item.name for item in fields(ProductRecord)}
    for index, row in frame.fillna("").iterrows():
        # Streamlit may reuse a deleted tail index for a newly added row.
        base = previous[index].to_dict() if 0 <= index < len(previous) and index not in deleted_rows else ProductRecord("manual", "人工新增").to_dict()
        updates = {key: value for key, value in row.to_dict().items() if key in valid_fields}
        editable = {"category", "product_name", "specification", "quantity_cartons", "original_grade_text", "price_marker"}
        base["manually_modified"] = bool(base.get("manually_modified")) or any(
            str(base.get(key) or "") != str(value or "") for key, value in updates.items() if key in editable
        )
        base.update(updates)
        base["warnings"] = [item for item in str(base.get("warnings", "")).split("；") if item]
        for name in ("quantity_cartons", "packing_quantity"):
            value = base.get(name)
            if pd.isna(value) or value == "":
                base[name] = None
            else:
                number = Decimal(str(value))
                base[name] = int(number) if number == number.to_integral_value() else number
        for name in ("unit_price", "total_price"):
            value = base.get(name)
            base[name] = None if pd.isna(value) or value == "" else Decimal(str(value))
        base["confidence"] = None if pd.isna(base.get("confidence")) or base.get("confidence") == "" else float(base["confidence"])
        base["source_location"] = base.get("source_location") or None
        result.append(ProductRecord(**{key: base.get(key) for key in valid_fields}))
    return result


@st.cache_resource(show_spinner=False, max_entries=1)
def local_ocr_extractor(master_path: str, fingerprint: str) -> LocalOCRExtractor:
    return LocalOCRExtractor(Path(master_path))


@st.cache_data(show_spinner=False)
def master_prices(master_path: str, fingerprint: str):
    prices, grade_markers, _ = load_master_data(master_path, fingerprint)
    return prices, grade_markers


try:
    master_path = discover_master_data(MASTER_DATA_DIR)
    master_fingerprint = master_data_fingerprint(master_path)
    prices, grade_markers = master_prices(str(master_path), master_fingerprint)
except (OSError, ValueError) as exc:
    diagnose("master_unavailable", dedupe="master_unavailable", error_type=type(exc).__name__, error=str(exc))
    st.error(f"主数据不可用，已阻断报价：{exc}")
    st.stop()


uploaded = st.file_uploader(
    "上传询价文件",
    type=["xlsx", "xls", "pdf", "png", "jpg", "jpeg", "webp", "bmp", "txt"],
    accept_multiple_files=True,
)
paste = st.text_area("或直接粘贴询价文字", height=140)
with st.expander("无需密钥：体验完整出单流程", expanded=True):
    st.code(DEMO_TEXT, language=None)
    st.caption("固定合成提案回放，不运行 AI、不读取上传内容；确认和出单仍使用完整业务校验。第三行演示缺包装时只显示小计。")
    if st.button("加载离线演示"):
        reset_task()
        st.session_state.pop("usage_diagnostics", None)
        try:
            demo_records, demo_audit = recognize_demo(master_path)
        except (InquiryError, ValueError, OSError):
            st.error("离线演示加载失败，请检查合成主数据与本地目录写入权限。")
        else:
            st.session_state.records = demo_records
            st.session_state.model_audit = demo_audit
            st.session_state.recognized = True
            st.rerun()
st.caption("真实 AI 识别会向 DeepSeek 官方 API 发送文字、OCR 结果和无价目录。原图仅在本地处理；原文中的联系信息不会自动脱敏。本地诊断会保留原附件、文字和响应。每个来源最多两次请求，无金额硬上限。")
external_consent = st.checkbox("我已检查本次内容，允许发送到 DeepSeek 并使用 API 额度", key="external_consent")
extractor = local_ocr_extractor(str(master_path), master_fingerprint)
show_diagnostic_warning()

if st.session_state.get("matched") and st.session_state.get("matched_fingerprint") != master_fingerprint:
    previous_fingerprint = st.session_state.get("matched_fingerprint")
    for record in st.session_state.records:
        match_price(record, prices, master_fingerprint)
    st.session_state.matched_fingerprint = master_fingerprint
    st.session_state.pop("generated", None)
    st.session_state.pop("price_editor", None)
    diagnose("master_changed", previous_fingerprint=previous_fingerprint, master_fingerprint=master_fingerprint,
             records=snapshot(st.session_state.records))
    st.warning("主数据内容已变化：旧价格、包装、总价、来源和生成状态已失效，并已使用当前主数据重新匹配。")

if uploaded:
    with st.expander("原始文件预览", expanded=False):
        for index, item in enumerate(uploaded):
            st.write(item.name)
            if Path(item.name).suffix.lower().lstrip(".") in IMAGE_SUFFIXES:
                try:
                    st.image(validate_image(item.getvalue()), width=700)
                except (OSError, ValueError) as error:
                    st.error(f"{item.name} 无法预览：{error}。请删除该文件后重新上传静态图片。")
            else:
                st.download_button(f"查看/下载 {item.name}", item.getvalue(), file_name=item.name, key=f"preview-{index}-{item.name}")

if st.button("识别新询价", type="primary", disabled=not external_consent):
    reset_task()
    diagnostic = Diagnostics()
    st.session_state.usage_diagnostics = diagnostic
    files, input_errors = [], []
    for item in uploaded or []:
        try:
            files.append((item.name, item.getvalue()))
        except Exception as error:
            input_errors.append({"source_name": item.name, "error_type": type(error).__name__, "error": str(error)})
    diagnostic.start(files, paste, version=APP_VERSION, master_fingerprint=master_fingerprint)
    for error in input_errors:
        diagnose("upload_read_failed", **error)
    show_diagnostic_warning()
    with st.status("正在准备识别…", expanded=False) as progress:
        try:
            if input_errors:
                raise InquiryError("上传原始文件不可读取，请重新上传；本次未调用模型")
            records, audit = recognize_inquiry(
                files, paste, master_path, extractor,
                output_directory=diagnostic.path / "model" if diagnostic.ready else None,
                on_stage=lambda label: progress.update(label=label),
                on_diagnostic=diagnostic.callback,
            )
            st.session_state.records = records
            st.session_state.model_audit = audit
            st.session_state.recognized = True
            failed = any(sample["status"] not in {"proposal_valid", "unresolved"} for sample in audit["run"]["samples"])
            st.session_state.recognition_status = {
                "label": "识别未完成，请处理提示" if failed else "识别完成，存在待核对项" if audit["notices"] else "识别完成，请核对结果",
                "state": "error" if failed else "complete",
            }
            diagnose("model_proposal", records=snapshot(records), notices=audit["notices"],
                     status=st.session_state.recognition_status, master_fingerprint=master_fingerprint)
        except InquiryError as error:
            diagnose("recognition_failed", error_type=type(error).__name__, error=str(error))
            st.session_state.recognition_error = str(error)
            st.session_state.recognition_status = {"label": "识别失败，请处理提示", "state": "error"}
        progress.update(**st.session_state.recognition_status)
    st.rerun()

if st.session_state.get("recognition_status"):
    st.status(**st.session_state.recognition_status, expanded=False)

if st.session_state.get("recognition_error"):
    st.error(st.session_state.recognition_error)

if st.session_state.get("recognized"):
    st.subheader("1. 核对识别结果")
    records = st.session_state.get("records", [])
    columns = ["category", "product_name", "specification", "quantity_cartons", "price_marker"]
    frame = pd.DataFrame([record.to_dict() for record in records], columns=columns)
    frame["category"] = frame["category"].astype(object)
    review_count = sum(
        bool(record.warnings)
        for record in records
    )
    if review_count:
        st.warning(f"{review_count} 行含核对提示；请逐行对照原始证据后再确认。")
    audit = st.session_state.get("model_audit", {})
    for notice in audit.get("notices", []):
        st.warning(notice)
    if not records:
        st.warning("本次没有产品提案；请核对证据是否为询价，或在下表人工新增。")
    with st.expander("本次识别证据与逐行待核对项", expanded=bool(audit.get("notices"))):
        for document, sample in zip(audit.get("documents", []), audit.get("run", {}).get("samples", [])):
            st.write("来源：", document["source_name"])
            for unit in document["units"]:
                location = f"第 {unit['page']} 页" if "page" in unit else f"{unit['sheet']} 第 {unit['row']} 行" if "sheet" in unit else f"第 {unit.get('line', '')} 行"
                method = "本地 OCR" if unit.get("reading_method") == "local_ocr" else "PDF 文字" if unit.get("reading_method") == "pdf_text" else "原文"
                st.text(f"{unit['unit_id']} · {location} · {method}\n{unit['text']}")
            validation = sample.get("validation") or {}
            parsed = validation.get("parsed") if not validation.get("structure_errors") else None
            parsed = parsed or {}
            for coverage in parsed.get("coverage", []):
                st.write(f"{coverage['unit_id']} 核对说明：{coverage['reason']}")
        for index, record in enumerate(records, 1):
            st.write(f"第 {index} 行 · {record.source_name} · {record.source_location or ''}")
            st.write("档位原文：", record.original_grade_text or "未提供", "；提案档位：", record.price_marker or "未决")
            for warning in record.warnings:
                st.write(warning)
    editor_key = f"recognition_editor-{st.session_state.get('recognition_editor_version', 0)}"
    edited = st.data_editor(
        frame,
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        placeholder="",
        column_config={
            "category": st.column_config.SelectboxColumn("分类", options=["沟槽", "玛钢", "球铁"], required=True),
            "product_name": st.column_config.TextColumn("产品名"),
            "specification": st.column_config.TextColumn("规格"),
            "quantity_cartons": st.column_config.NumberColumn("箱数", min_value=1, step=1),
            "price_marker": st.column_config.SelectboxColumn("等级", options=["A", "a", "b"]),
        },
        key=editor_key,
    )
    editor_state = st.session_state[editor_key]
    preview = records_from_frame(edited, records, editor_state.get("deleted_rows", []))
    confirmation_fields = ("category", "product_name", "specification", "quantity_cartons", "original_grade_text", "price_marker")
    recognition_changed = bool(editor_state.get("deleted_rows") or editor_state.get("added_rows")) or len(records) != len(preview) or any(
        any(getattr(before, field) != getattr(after, field) for field in confirmation_fields)
        for before, after in zip(records, preview)
    )
    diagnostic = st.session_state.get("usage_diagnostics")
    if any(editor_state.get(key) for key in ("edited_rows", "deleted_rows", "added_rows")) or diagnostic and editor_key in diagnostic._last:
        diagnose("recognition_edited", dedupe=editor_key, dedupe_value=editor_state, editor_key=editor_key, baseline=snapshot(records),
                 delta=editor_state, records=snapshot(preview))
    if st.session_state.get("recognition_confirmed") and recognition_changed:
        # Keep the original rows as the editor's delta baseline until confirmation.
        for record in records + preview:
            record.manually_confirmed = False
            record.confirmed_inquiry = None
            match_price(record, prices, master_fingerprint)
        st.session_state.recognition_confirmed = False
        st.session_state.matched = False
        st.session_state.pop("generated", None)
        st.session_state.pop("price_editor", None)
        st.warning("分类、产品名、规格、箱数或等级已修改；旧匹配已失效，请重新完成本次人工确认。")
    if st.button("确认识别结果"):
        current = preview
        confirmation_input = snapshot(current)
        errors = []
        for index, record in enumerate(current, 1):
            if record.category not in {"沟槽", "玛钢", "球铁"}:
                categories = required_categories(record, prices)
                suffix = f"（当前名称/规格候选分类：{'、'.join(sorted(categories))}）" if categories else ""
                errors.append(f"第 {index} 行必须选择沟槽、玛钢或球铁分类{suffix}")
            if not record.product_name:
                errors.append(f"第 {index} 行缺少产品名")
            if not record.specification:
                errors.append(f"第 {index} 行缺少规格")
            try:
                record.quantity_cartons = positive_integer(record.quantity_cartons, "箱数")
            except ValueError as exc:
                errors.append(f"第 {index} 行：{exc}")
            if (record.original_grade_text or record.grade_review_required) and not record.price_marker:
                errors.append(f"第 {index} 行等级/a-A-b 有歧义，必须人工选择")
        if not errors:
            for index, record in enumerate(current, 1):
                try:
                    confirm_inquiry(record)
                except ValueError as exc:
                    errors.append(f"第 {index} 行：{exc}")
        if errors:
            diagnose("confirmation_failed", records=snapshot(current), submitted=confirmation_input, errors=errors,
                     master_fingerprint=master_fingerprint)
            st.error("；".join(errors))
        else:
            diagnose("confirmation_succeeded", records=snapshot(current), submitted=confirmation_input,
                     master_fingerprint=master_fingerprint, meaning="user_confirmed_version")
            st.session_state.records = current
            st.session_state.recognition_confirmed = True
            st.session_state.matched = False
            st.session_state.pop("generated", None)
            reset_recognition_editor()
            st.rerun()

if st.session_state.get("recognition_confirmed"):
    st.success("识别结果已确认，可以匹配价格。")
    if st.button("匹配标准主数据"):
        for record in st.session_state.records:
            match_price(record, prices, master_fingerprint)
        st.session_state.matched = True
        st.session_state.matched_fingerprint = master_fingerprint
        st.session_state.pop("generated", None)
        st.session_state.pop("price_editor", None)
        diagnose("matched", records=snapshot(st.session_state.records), master_fingerprint=master_fingerprint)
        st.rerun()

if st.session_state.get("matched"):
    st.subheader("2. 匹配结果")
    records = st.session_state.records
    price_columns = ["product_name", "specification", "price_marker"]
    frame = pd.DataFrame([record.to_dict() for record in records], columns=price_columns)
    edited_prices = st.data_editor(
        frame,
        num_rows="fixed",
        hide_index=True,
        width="stretch",
        column_config={
            "product_name": st.column_config.TextColumn("产品名"),
            "specification": st.column_config.TextColumn("规格"),
            "price_marker": st.column_config.SelectboxColumn("a/A/b", options=["a", "A", "b"], required=True),
        },
        disabled=["product_name", "specification"],
        key="price_editor",
    )
    current = records_from_frame(edited_prices, records)
    changed_markers = [
        record
        for before, record in zip(records, current)
        if before.price_marker != record.price_marker
    ]
    if changed_markers:
        before_markers = snapshot(records)
        for record in changed_markers:
            confirm_inquiry(record)
            match_price(record, prices, master_fingerprint)
        st.session_state.records = current
        st.session_state.matched_fingerprint = master_fingerprint
        st.session_state.pop("generated", None)
        diagnose("grade_changed", before=before_markers, records=snapshot(current), master_fingerprint=master_fingerprint)
        st.rerun()
    errors = []
    pending_price_count = 0
    for index, record in enumerate(current, 1):
        if record.match_status == "pending" and record.unit_price is None and record.price_source_file:
            pending_price_count += 1
            continue
        if not record.price_source_file:
            errors.append(f"第 {index} 行：{record.match_message or '没有主数据价格来源，必须补充主数据后重新匹配'}")
    missing_count = sum(
        record.unit_price is not None and record.total_price is None and bool(record.price_source_file)
        for record in current
    )
    quotable_count = sum(record.unit_price is not None and bool(record.price_source_file) for record in current)
    if pending_price_count:
        st.warning(f"{pending_price_count} 行面价待补充；这些行不计算金额且不会写入报价，其他有价行可继续生成。")
    if missing_count:
        st.info(f"{missing_count} 行无包装数，销售单包装、总个数及行金额将留空，整单不会显示完整总计。")
    if errors:
        st.error("；".join(errors))

    try:
        customer_names = load_customer_names(CUSTOMER_HISTORY_PATH)
        customer_history_ok = True
    except ValueError as exc:
        diagnose("customer_history_failed", dedupe="customer_history", error_type=type(exc).__name__, error=str(exc))
        st.error(f"单位名称历史不可用，已阻断生成：{exc}")
        customer_names = []
        customer_history_ok = False
    demo_mode = st.session_state.get("model_audit", {}).get("demo", False)
    if demo_mode and "示例客户（虚构）" not in customer_names:
        customer_names.append("示例客户（虚构）")
    customer_name = st.selectbox(
        "单位名称",
        customer_names,
        index=customer_names.index("示例客户（虚构）") if demo_mode else None,
        placeholder="搜索已有单位名称，或直接输入新名称",
        accept_new_options=True,
        filter_mode="fuzzy",
        on_change=clear_generated,
        key="customer_name",
    )
    customer_ready = customer_history_ok and bool(str(customer_name or "").strip())
    diagnose("customer_name", dedupe="customer_name", value=customer_name)

    preview_ready = False
    if not errors and quotable_count and customer_ready:
        st.subheader("3. 销售单预览")
        try:
            preview_pages = render_quotation_preview(current, customer_name, master_data_dir=MASTER_DATA_DIR)
            if not preview_pages:
                raise ValueError("未返回预览页面")
            for page_number, page in enumerate(preview_pages, 1):
                st.image(page, caption=f"第 {page_number} 页", width="stretch")
            preview_ready = True
            diagnose("preview_succeeded", dedupe="preview", customer_name=customer_name, records=snapshot(current),
                     master_fingerprint=master_fingerprint, page_count=len(preview_pages))
        except Exception as exc:
            diagnose("preview_failed", dedupe="preview", customer_name=customer_name, records=snapshot(current),
                     master_fingerprint=master_fingerprint, error_type=type(exc).__name__, error=str(exc))
            st.error(f"预览失败：{exc}")

    st.subheader("4. 生成销售单 PDF 与逐页 PNG")
    if st.button("生成销售单", type="primary", disabled=not preview_ready):
        diagnose("generation_started", records=snapshot(current), customer_name=customer_name, master_fingerprint=master_fingerprint)
        try:
            path, mapping = generate_quotation(
                current,
                OUTPUT_DIR,
                customer_name,
                master_data_dir=MASTER_DATA_DIR,
                customer_history_path=CUSTOMER_HISTORY_PATH,
            )
        except (OSError, ValueError) as exc:
            diagnose("generation_failed", error_type=type(exc).__name__, error=str(exc))
            st.error(f"生成失败：{exc}")
        else:
            diagnose("generation_returned", pdf=str(path), mapping=mapping)
            st.session_state.records = current
            st.session_state.generated = (path, mapping)
    if st.session_state.get("generated"):
        path, mapping = st.session_state.generated
        try:
            pdf_data = Path(path).read_bytes()
            png_data = [Path(png_path).read_bytes() for png_path in mapping["png_paths"]]
        except OSError as exc:
            diagnose("output_read_failed", pdf=str(path), mapping=mapping, error_type=type(exc).__name__, error=str(exc))
            st.session_state.pop("generated", None)
            st.error("输出文件不可读取，请重新生成。")
        else:
            diagnose("output_read_succeeded", dedupe="output_read", pdf=str(path), mapping=mapping,
                     pdf_sha256=sha256(pdf_data).hexdigest(), png_sha256=[sha256(data).hexdigest() for data in png_data],
                     pdf_bytes=len(pdf_data), png_bytes=[len(data) for data in png_data])
            st.success(
                f"已生成销单 {mapping['sales_id']}：PDF 1 份、PNG {mapping['page_count']} 张；"
                f"{mapping['total_label']} ¥{mapping['total']:.2f}"
            )
            st.download_button("下载 PDF", pdf_data, file_name=Path(path).name)
            for page_number, (png_path, data) in enumerate(zip(mapping["png_paths"], png_data), 1):
                st.download_button(
                    f"下载第 {page_number} 页 PNG",
                    data,
                    file_name=Path(png_path).name,
                    key=f"download-png-{mapping['sales_id']}-{page_number}",
                )

diagnostic = st.session_state.get("usage_diagnostics")
if diagnostic and diagnostic.ready:
    bundle = st.session_state.get("diagnostic_zip")
    if not bundle or bundle[0] != diagnostic.revision:
        if st.button("准备本次诊断包"):
            data = diagnostic.package()
            if data is not None:
                st.session_state.diagnostic_zip = (diagnostic.revision, data)
                st.rerun()
    else:
        st.download_button("下载本次诊断包", bundle[1], file_name=f"diagnostics-{diagnostic.id}.zip", mime="application/zip")
show_diagnostic_warning()
