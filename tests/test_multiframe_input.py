from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from src import agent_inquiry as inquiry
from src.input_router import extract_evidence_units


MASTER = inquiry.ROOT / "主数据"


def _image_bytes(format_name, *, frames=2):
    images = [Image.new("RGB", (20, 20), color) for color in ("white", "black")[:frames]]
    output = BytesIO()
    images[0].save(output, format_name, save_all=frames > 1, append_images=images[1:])
    return output.getvalue()


@pytest.mark.parametrize("filename,format_name", [("frames.png", "PNG"), ("frames.webp", "WEBP")])
def test_multiframe_images_are_rejected_before_ocr_and_model(tmp_path, filename, format_name):
    ocr_calls = []
    transport_calls = []
    ocr = SimpleNamespace(extract_raw=lambda data: ocr_calls.append(data))

    with pytest.raises(inquiry.InquiryError) as error:
        inquiry.recognize_inquiry(
            [(filename, _image_bytes(format_name))], "", MASTER, ocr,
            transport=lambda *args, **kwargs: transport_calls.append((args, kwargs)),
            environ={"DEEPSEEK_API_KEY": "secret-must-not-leak"}, output_root=tmp_path,
        )

    assert "检测到多帧图片，请拆分成静态图片后重传" in str(error.value)
    assert "secret-must-not-leak" not in str(error.value)
    assert ocr_calls == []
    assert transport_calls == []


def test_single_frame_image_still_reaches_local_ocr():
    ocr_calls = []
    raw = [{"res": {"parsing_res_list": [{"block_label": "text", "block_content": "单帧图片"}]}}]
    ocr = SimpleNamespace(extract_raw=lambda data: ocr_calls.append(data) or raw)

    units = extract_evidence_units("single.png", _image_bytes("PNG", frames=1), ocr)

    assert len(ocr_calls) == 1
    assert units[0]["text"] == "单帧图片"
