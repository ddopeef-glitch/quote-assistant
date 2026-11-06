from pathlib import Path

import fitz
import pytest
from streamlit.testing.v1 import AppTest

from config import MASTER_DATA_FILE
from src import core, agent_inquiry
from src.demo import recognize_demo
from scripts.create_demo_data import make_book


def button(app, label):
    return next(item for item in app.button if item.label == label)


def test_setup_never_replaces_an_existing_workbook(tmp_path):
    path = tmp_path / "master.xlsx"
    path.write_bytes(b"user-owned")
    with pytest.raises(FileExistsError):
        make_book(path)
    assert path.read_bytes() == b"user-owned"


def test_filled_failure_placeholder_cannot_be_confirmed_or_priced():
    item = core.ProductRecord("manual", "synthetic", category="沟槽", product_name="90°弯头",
                              specification="108", quantity_cartons=2, price_marker="b",
                              extraction_method="待人工处理")
    with pytest.raises(ValueError, match="仅填写五列不能确认"):
        core.confirm_inquiry(item)
    prices, _, fingerprint = core.load_master_data(MASTER_DATA_FILE)
    assert core.match_price(item, prices, fingerprint) == []
    assert item.total_price is None and not item.manually_confirmed
    manual = core.ProductRecord("manual", "人工新增", category="沟槽", product_name="90°弯头",
                                specification="108", quantity_cartons=2, price_marker="b")
    core.confirm_inquiry(manual)
    assert len(core.match_price(manual, prices, fingerprint)) == 1
    assert manual.total_price == 240


def test_offline_demo_validates_without_credentials_and_requires_confirmation(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    records, audit = recognize_demo(MASTER_DATA_FILE, output_root=tmp_path)
    assert len(records) == 3 and audit["demo"] is True
    assert audit["run"]["samples"][0]["status"] == "proposal_valid"
    assert all(not item.manually_confirmed and item.unit_price is None for item in records)
    prices, _, fingerprint = core.load_master_data(MASTER_DATA_FILE)
    for item in records:
        assert core.match_price(item, prices, fingerprint) == []
        core.confirm_inquiry(item)
        assert len(core.match_price(item, prices, fingerprint)) == 1
    assert [item.total_price for item in records] == [240, 660, None]
    assert records[-1].packing_quantity is None


def test_bundled_font_can_render_and_reopen_chinese_pdf(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "PDF_FONT_PATH", tmp_path / "absent.ttf")
    monkeypatch.setattr(core, "PDF_BOLD_FONT_PATH", tmp_path / "absent-bold.ttf")
    doc = fitz.open()
    page = doc.new_page()
    core._draw_text(page, fitz.Rect(30, 30, 500, 80), "虚构演示销售单 123.45", 17, bold=True)
    with fitz.open(stream=doc.tobytes(), filetype="pdf") as reopened:
        assert "虚构演示销售单" in reopened[0].get_text()
        assert "123.45" in reopened[0].get_text()
    doc.close()


def test_ui_demo_to_real_pdf_without_external_consent(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(agent_inquiry, "RUNS_DIR", tmp_path / "audit")
    output = tmp_path / "output"
    history = tmp_path / "settings/customer_names.json"
    monkeypatch.setattr(core, "OUTPUT_ROOT", output)
    monkeypatch.setattr("config.OUTPUT_DIR", output)
    monkeypatch.setattr(core, "CUSTOMER_HISTORY_PATH", history)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=30).run()
    assert not app.exception
    assert button(app, "识别新询价").disabled
    button(app, "加载离线演示").click().run()
    assert not app.exception and not history.exists()
    assert all(not record.manually_confirmed for record in app.session_state["records"])
    button(app, "确认识别结果").click().run()
    button(app, "匹配标准主数据").click().run()
    assert not app.exception and not button(app, "生成销售单").disabled
    assert not output.exists() and not history.exists()
    button(app, "生成销售单").click().run()
    assert not app.exception
    assert not app.error, [error.value for error in app.error]
    pdf, mapping = app.session_state["generated"]
    assert pdf.is_file() and len(mapping["png_paths"]) == 1
    assert mapping["total"] == 900 and mapping["total_label"] == "已计算小计"
    with fitz.open(pdf) as document:
        text = document[0].get_text()
        assert "虚构数据" in text and "900.00" in text and "已计算小计" in text
    assert core.load_customer_names(history) == ["示例客户（虚构）"]
    assert button(app, "识别新询价").disabled
