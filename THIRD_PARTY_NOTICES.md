# 第三方组件与许可

作者已确认本项目源码为本人编写。本脱敏公开版尚未选择代码开源许可证；不要将缺少 LICENSE 误写为 MIT 或“可自由商用”。

基础依赖通过 pip 安装，本源码包不附带第三方运行库或模型权重。各组件以其自身许可为准，包括 Streamlit、pandas、openpyxl、Pillow、tzdata、pytest 和 PyMuPDF。

PyMuPDF / MuPDF 提供 AGPL 和商业许可选项，具体使用与分发条件应对照 [官方许可说明](https://pymupdf.readthedocs.io/en/latest/about.html#license-and-copyright)。本文件不将整个项目宣称为任一开源许可证，也不认定商业部署已满足许可条件。

Windows 使用机器已有的中文字体；其他环境回退到 PyMuPDF 自带字体。没有复制系统字体文件。可选 OCR 的 PaddleOCR、PaddlePaddle 及模型权重需分别核对上游许可与运行条件。

接口与字体实现参考：[DeepSeek thinking mode](https://api-docs.deepseek.com/guides/thinking_mode/)、[PyMuPDF Font](https://pymupdf.readthedocs.io/en/latest/font.html)、[PaddleOCR-VL 安装与使用](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PaddleOCR-VL.html)。链接核对日期：2026-11-05；真实接口兼容性与模型效果未在本轮重新验收。
