## 🌟 Introduction

YomiToku is a Document AI engine specialized in Japanese document image analysis. It provides full OCR (optical character recognition) and layout analysis capabilities, enabling the recognition, extraction, and conversion of text and diagrams from images.

- 🤖 Equipped with four AI models trained on Japanese datasets: text detection, text recognition, layout analysis, and table structure recognition. All models are independently trained and optimized for Japanese documents, delivering high-precision inference.
- 🇯🇵 Each model is specifically trained for Japanese document images, supporting the recognition of over 7,000 Japanese characters, including vertical text and other layout structures unique to Japanese documents. (It also supports English documents.)
- 📈 By leveraging layout analysis, table structure parsing, and reading order estimation, it extracts information while preserving the semantic structure of the document layout.
- 📄 Supports a variety of output formats, including HTML, Markdown, JSON, and CSV. It also allows for the extraction of diagrams and images contained within the documents.It also supports converting document images into fully text-searchable PDFs.
- ⚡ Operates efficiently in GPU environments, enabling fast document transcription and analysis. It requires less than 8GB of VRAM, eliminating the need for high-end GPUs.。

## 🙋 FAQ

### Q. Is it possible to use YomiToku in an environment without internet access?

A. Yes, it is possible.
YomiToku connects to Hugging Face Hub to automatically download model files during the first execution, requiring internet access at that time. However, you can manually download the files in advance, allowing YomiToku to operate in an offline environment. For details, please refer to [Module Usage](module.en.md) under the section "Using YomiToku in an Offline Environment."

### Q. Is commercial use allowed?

A. This package is licensed under CC BY-NC 4.0. It is available for free for personal and research purposes. For commercial use, a paid commercial license is required. Please contact the developers for further details.
