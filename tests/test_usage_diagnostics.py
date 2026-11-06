"""Local diagnostics acceptance using synthetic inputs and offline UI transports."""
from io import BytesIO
import json
import os
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import fitz
from PIL import Image
import pytest
from streamlit.testing.v1 import AppTest

from src import agent_inquiry as inquiry, agent_workflow as workflow, core
from src.ocr_pipeline import LocalOCRExtractor
from src.input_router import extract_evidence_units
from src.usage_diagnostics import Diagnostics, code_fingerprint
from test_agent_inquiry import MASTER, TEXT, edit_recognition, model_transport


def events(diagnostic):
    return [json.loads(line) for line in (diagnostic.path / "events.jsonl").read_text(encoding="utf-8").splitlines()]


def button(app, label):
    return next(button for button in app.button if button.label == label)


def start(monkeypatch, *, image=False):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-secret")
    monkeypatch.setattr(workflow, "stream_transport", model_transport([]))
    if image:
        import streamlit as st
        buffer = BytesIO()
        Image.new("RGB", (20, 20), "white").save(buffer, "PNG")
        monkeypatch.setattr(st, "file_uploader", lambda *a, **k: [SimpleNamespace(
            name="synthetic.png", type="image/png", getvalue=buffer.getvalue)])
        monkeypatch.setattr(LocalOCRExtractor, "extract_raw", lambda *a: [{"res": {
            "parsing_res_list": [{"block_content": TEXT}]}}])
    app = AppTest.from_file(str(inquiry.ROOT / "app.py"), default_timeout=20).run()
    app.checkbox(key="external_consent").check().run()
    if not image:
        app.text_area[0].set_value(TEXT)
    button(app, "识别新询价").click().run()
    assert not app.exception
    return app


def test_original_bytes_mapping_exclusive_runs_and_portable_fingerprint(tmp_path):
    diagnostic = Diagnostics(root=tmp_path / "records")
    diagnostic.start([("../../same.png", b"one"), ("../../same.png", b"two"), ("C:\\evil\\x.txt", b"three")],
                     " \n粘贴原文\n", version="2", master_fingerprint="master")
    meta = json.loads((diagnostic.path / "meta.json").read_text(encoding="utf-8"))
    assert [item["source_id"] for item in meta["originals"]] == ["source-4", "source-1", "source-2", "source-3"]
    assert [item["original_name"] for item in meta["originals"]][1:3] == ["../../same.png"] * 2
    assert [(diagnostic.path / item["file"]).read_bytes() for item in meta["originals"]] == [" \n粘贴原文\n".encode(), b"one", b"two", b"three"]
    assert not diagnostic.errors
    other = Diagnostics(root=diagnostic.root)
    other.start([], "", version="2", master_fingerprint="master")
    assert other.path != diagnostic.path and diagnostic.path.exists()
    for name in ("first", "second"):
        root = tmp_path / name
        (root / "src").mkdir(parents=True)
        for file in ("app.py", "config.py", "VERSION", "requirements.txt", "src/a.py"):
            (root / file).write_text(file)
    assert code_fingerprint(tmp_path / "first") == code_fingerprint(tmp_path / "second")
    (tmp_path / "second" / "src/a.py").write_text("changed")
    assert code_fingerprint(tmp_path / "first") != code_fingerprint(tmp_path / "second")


def test_secrets_never_break_json_and_failures_remain_visible(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", 'actual-"secret')
    diagnostic = Diagnostics(root=tmp_path / "records")
    diagnostic.start([], "original", version="2", master_fingerprint="master")
    diagnostic.event("ocr_returned", raw={"Authorization": "Bearer arbitrary-token", "api_key": "other-secret",
                     "message": 'password=ab"cd actual-"secret', "content": 'quoted "evidence"'})
    diagnostic.failure("OCR", ValueError('Authorization: Bearer live-token actual-"secret'))
    text = (diagnostic.path / "events.jsonl").read_text(encoding="utf-8")
    assert all(word not in text for word in ("arbitrary-token", "other-secret", "live-token", "actual-"))
    assert events(diagnostic)[-1]["event"] == "diagnostic_failed"
    assert events(diagnostic)[-2]["data"]["raw"]["content"] == 'quoted "evidence"'
    diagnostic.start([], 'actual-"secret', version="2", master_fingerprint="master")
    assert any("FileExistsError" in error for error in diagnostic.errors)
    assert (diagnostic.path / "input-000.txt").read_text() == "original"


@pytest.mark.parametrize("raw", [{"unexpected": "raw OCR"}, [{"res": {"bad": "raw OCR"}}]])
def test_partial_document_and_source_evidence_survives_failed_ocr(tmp_path, monkeypatch, raw):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-secret")
    with fitz.open() as pdf:
        pdf.new_page().insert_text((30, 40), "first successful page")
        pdf.new_page()
        data = pdf.tobytes()
    diagnostic = Diagnostics(root=tmp_path / "records")
    files = [("good.txt", b"successful source"), ("partial.pdf", data)]
    diagnostic.start(files, "", version="2", master_fingerprint="master")
    with pytest.raises(inquiry.InquiryError, match="读取失败"):
        inquiry.recognize_inquiry(files, "", MASTER, SimpleNamespace(extract_raw=lambda data: raw),
            transport=lambda *a, **k: pytest.fail("failed inputs must not call model"),
            on_diagnostic=diagnostic.callback)
    rows = events(diagnostic)
    assert next(row for row in rows if row["event"] == "ocr_returned")["data"]["raw"] == raw
    assert [row["data"]["unit"]["text"].strip() for row in rows if row["event"] == "evidence_unit"] == ["successful source", "first successful page"]
    failure = next(row["data"] for row in rows if row["event"] == "source_failed")
    assert failure["source_id"] == "source-2" and failure["error_type"] == "ValueError"
    assert not any(row["event"] == "workflow_started" for row in rows)
    assert extract_evidence_units("a.txt", b"unchanged", on_diagnostic=lambda *a: 1 / 0)[0]["text"] == "unchanged"


def test_full_ui_evidence_edit_confirm_match_preview_generate_and_rerun(tmp_path, monkeypatch):
    buffer = BytesIO()
    Image.new("RGB", (20, 20), "white").save(buffer, "PNG")
    monkeypatch.setattr(core, "load_customer_names", lambda *a: ["合成测试单位"])
    monkeypatch.setattr(core, "render_quotation_preview", lambda *a, **k: [buffer.getvalue()])
    pdf, png = tmp_path / "synthetic.pdf", tmp_path / "synthetic.png"
    def generate(*args, **kwargs):
        pdf.write_bytes(b"synthetic pdf")
        png.write_bytes(buffer.getvalue())
        return pdf, {"sales_id": "QA-SYNTH-TESTS", "page_count": 1, "total_label": "合计", "total": 1, "png_paths": [png]}
    monkeypatch.setattr(core, "generate_quotation", generate)
    app = start(monkeypatch, image=True)
    diagnostic = app.session_state["usage_diagnostics"]
    edit_recognition(app, edited={0: {"product_name": ""}}, confirm=True)
    assert any(row["event"] == "confirmation_failed" for row in events(diagnostic))
    edit_recognition(app, edited={})  # Reverting to the untouched baseline is an observed edit too.
    assert [row for row in events(diagnostic) if row["event"] == "recognition_edited"][-1]["data"]["delta"]["edited_rows"] == {}
    new_row = {"category": "沟槽", "product_name": "90°弯头", "specification": "108", "quantity_cartons": 2}
    edit_recognition(app, deleted=[0], added=[new_row], confirm=True)
    assert not app.exception and app.session_state["recognition_confirmed"]
    button(app, "匹配标准主数据").click().run()
    app.selectbox[0].select("合成测试单位").run()
    button(app, "生成销售单").click().run()
    assert not app.exception
    initial_count = len(events(diagnostic))
    app.run()
    assert len(events(diagnostic)) == initial_count
    names = [row["event"] for row in events(diagnostic)]
    for expected in ("recognition_started", "ocr_returned", "source_read", "workflow_started", "workflow_returned",
                     "model_proposal", "recognition_edited", "confirmation_succeeded", "matched", "customer_name",
                     "preview_succeeded", "generation_started", "generation_returned", "output_read_succeeded"):
        assert expected in names
    edit = [row["data"] for row in events(diagnostic) if row["event"] == "recognition_edited"][-1]
    assert edit["baseline"][0]["source_name"] == "synthetic.png"
    assert edit["delta"]["deleted_rows"] == [0] and edit["records"][0]["source_type"] == "manual"
    assert app.session_state["records"][0].original_text == ""
    app.session_state["matched_fingerprint"] = "older-master"
    app.run()
    assert events(diagnostic)[-1]["event"] == "master_changed"
    model_dir = Path(next(row["data"]["evidence_directory"] for row in events(diagnostic) if row["event"] == "workflow_started"))
    assert model_dir == diagnostic.path / "model"
    assert "离线测试推理" in (model_dir / "source-1/round-02/response.json").read_text(encoding="utf-8")
    button(app, "准备本次诊断包").click().run()
    assert any(item.label == "下载本次诊断包" for item in app.get("download_button"))
    with ZipFile(BytesIO(app.session_state["diagnostic_zip"][1])) as archive:
        assert {"meta.json", "events.jsonl", "input-001.png"} <= set(archive.namelist())
        assert any(name.endswith("response.json") for name in archive.namelist())
        assert not any(name.endswith("synthetic.pdf") or name.endswith("synthetic.png") for name in archive.namelist())
    old_path = diagnostic.path
    button(app, "识别新询价").click().run()
    assert old_path.exists() and app.session_state["usage_diagnostics"].path != old_path


def test_ui_grade_edit_is_recorded_in_diagnostics(monkeypatch):
    app = start(monkeypatch)
    diagnostic = app.session_state["usage_diagnostics"]
    edit_recognition(app, edited={0: {"price_marker": "A"}})
    edit = [row["data"] for row in events(diagnostic) if row["event"] == "recognition_edited"][-1]
    assert next(iter(edit["delta"]["edited_rows"].values()))["price_marker"] == "A"
    assert edit["records"][0]["price_marker"] == "A"


def test_empty_input_and_missing_credentials_still_get_separate_records(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(workflow, "stream_transport", lambda *a, **k: pytest.fail("must not call provider"))
    app = AppTest.from_file(str(inquiry.ROOT / "app.py"), default_timeout=20).run()
    app.checkbox(key="external_consent").check().run()
    button(app, "识别新询价").click().run()
    first = app.session_state["usage_diagnostics"]
    assert events(first)[-2]["data"]["reason"] == "empty_input"
    app.text_area[0].set_value(TEXT)
    button(app, "识别新询价").click().run()
    second = app.session_state["usage_diagnostics"]
    assert second.path != first.path and events(second)[-2]["data"]["reason"] == "unavailable_credentials"
    assert (second.path / "input-000.txt").read_text(encoding="utf-8") == TEXT


def test_new_recording_write_failure_does_not_block_confirmation_or_matching(monkeypatch):
    app = start(monkeypatch)
    diagnostic = app.session_state["usage_diagnostics"]
    monkeypatch.setattr(diagnostic, "_append", lambda *a: (_ for _ in ()).throw(PermissionError("password=private")))
    button(app, "确认识别结果").click().run()
    button(app, "匹配标准主数据").click().run()
    assert not app.exception and app.session_state["matched"]
    assert any("诊断记录不完整" in warning.value for warning in app.warning)
    assert "private" not in "".join(diagnostic.errors)
    app.run()
    assert any("诊断记录不完整" in warning.value for warning in app.warning)


def test_initialization_failure_uses_existing_model_audit_and_warns(tmp_path, monkeypatch):
    from src import usage_diagnostics
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("preserve")
    monkeypatch.setattr(usage_diagnostics, "DIAGNOSTICS_DIR", blocked / "records")
    monkeypatch.setattr(inquiry, "RUNS_DIR", tmp_path / "existing-model-audit")
    app = start(monkeypatch)
    assert app.session_state["recognized"] and not app.session_state["usage_diagnostics"].ready
    assert any("诊断记录不完整" in warning.value for warning in app.warning)
    assert Path(app.session_state["model_audit"]["run"]["evidence_directory"]).is_relative_to(tmp_path / "existing-model-audit")
    assert blocked.read_text() == "preserve"


def test_ui_grade_preview_generation_and_output_failures_are_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "load_customer_names", lambda *a: ["合成测试单位"])
    buffer = BytesIO()
    Image.new("RGB", (20, 20), "white").save(buffer, "PNG")
    mode = ["preview_failed"]
    def preview(*args, **kwargs):
        if mode[0] == "preview_failed":
            raise ValueError("synthetic preview failure")
        return [buffer.getvalue()]
    def generate(*args, **kwargs):
        if mode[0] == "generation_failed":
            raise OSError("synthetic generation failure")
        return tmp_path / "missing.pdf", {"png_paths": [tmp_path / "missing.png"]}
    monkeypatch.setattr(core, "render_quotation_preview", preview)
    monkeypatch.setattr(core, "generate_quotation", generate)
    app = start(monkeypatch)
    diagnostic = app.session_state["usage_diagnostics"]
    button(app, "确认识别结果").click().run()
    button(app, "匹配标准主数据").click().run()
    for marker in ("a", "b"):
        states = app._tree.get_widget_states()
        states.widgets.add(id=app.dataframe[1].proto.id, string_value=json.dumps({
            "deleted_rows": [], "added_rows": [], "edited_rows": {0: {"price_marker": marker}}}))
        app._run(widget_state=states)
        assert not app.exception
    assert len([row for row in events(diagnostic) if row["event"] == "grade_changed"]) == 2
    app.selectbox[0].select("合成测试单位").run()
    assert events(diagnostic)[-1]["event"] == "preview_failed"
    revision = diagnostic.revision
    app.run()
    assert diagnostic.revision == revision
    mode[0] = "generation_failed"
    app.run()
    button(app, "生成销售单").click().run()
    assert events(diagnostic)[-1]["event"] == "generation_failed"
    mode[0] = "read_failed"
    button(app, "生成销售单").click().run()
    assert events(diagnostic)[-1]["event"] == "output_read_failed"
    assert app.session_state["matched"] and app.session_state["recognition_confirmed"]
    assert "generated" not in app.session_state


def test_package_rejects_junction_and_regular_file_errors(tmp_path, monkeypatch):
    diagnostic = Diagnostics(root=tmp_path / "records")
    diagnostic.start([], "safe", version="2", master_fingerprint="master")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("do not read")
    link = diagnostic.path / "linked"
    if os.name == "nt":
        import _winapi
        _winapi.CreateJunction(str(outside), str(link))
    else:
        link.symlink_to(outside, target_is_directory=True)
    try:
        assert diagnostic.package() is None and diagnostic.errors
        assert (outside / "secret.txt").read_text() == "do not read"
    finally:
        os.rmdir(link) if os.name == "nt" else link.unlink()
    assert diagnostic.package() is not None
    monkeypatch.setattr(os, "walk", lambda *args, **kwargs: kwargs["onerror"](PermissionError("unreadable model directory")))
    assert diagnostic.package() is None
    assert any("unreadable model directory" in error for error in diagnostic.errors)
