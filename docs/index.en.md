# 🌟 Introduction

YomiToku is a Document AI engine specialized in Japanese document image analysis. It provides full OCR (optical character recognition) and layout analysis capabilities, enabling the recognition, extraction, and conversion of text and diagrams from images.

- 🤖 Equipped with four AI models trained on Japanese datasets: text detection, text recognition, layout analysis, and table structure recognition. All models are independently trained and optimized for Japanese documents, delivering high-precision inference.
- 🇯🇵 Each model is specifically trained for Japanese document images, supporting the recognition of over 7,000 Japanese characters, including vertical text and other layout structures unique to Japanese documents. (It also supports English documents.)
- 📈 By leveraging layout analysis, table structure parsing, and reading order estimation, it extracts information while preserving the semantic structure of the document layout.
- 📄 Supports a variety of output formats, including HTML, Markdown, JSON, and CSV. It also allows for the extraction of diagrams and images contained within the documents.It also supports converting document images into fully text-searchable PDFs.
- ⚡ Operates efficiently in GPU environments, enabling fast document transcription and analysis. It requires less than 8GB of VRAM, eliminating the need for high-end GPUs. In efficient mode, fast inference is possible even on a CPU.

## 🙋 FAQ

### Q. Is it possible to use YomiToku in an environment without internet access?

A. Yes, it is possible.
YomiToku connects to Hugging Face Hub to automatically download model files during the first execution, requiring internet access at that time. However, you can manually download the files in advance, allowing YomiToku to operate in an offline environment. For details, please refer to [Module Usage](module.en.md) under the section "Using YomiToku in an Offline Environment."

### Q. Is commercial use allowed?

Yes. For commercial use of YomiToku, we provide a **licensed product edition** through the options below.  
The commercial edition includes numerous enhancements such as improved handwriting recognition accuracy, automatic image orientation correction, and advanced layout analysis features that are **available only in the product version**.

- **[Guideline for Determining Commercial / Non-Commercial Use](commercial_use_guideline.en.md)**

#### On-Premises / Local PC Commercial Use

If you wish to use YomiToku commercially in an on-premises environment or on a local PC,  
we offer a dedicated **on-premises commercial license**.  
For more details, please contact us via:

- <https://www.mlism.com/>

#### Cloud-Based Commercial Use (AWS Marketplace)

The commercial edition of YomiToku is also available on **AWS Marketplace**.  
All processing is executed **entirely within your own AWS environment**, with no external network communication or transmission to third-party servers.  
This ensures safe usage even for workloads involving confidential documents, internal corporate materials, or personal information.

The following products are available. Each is a separate AWS Marketplace product, so subscribe to each product you plan to use.

| Product | Description |
| :--- | :--- |
| [YomiToku-Pro - Document Analyzer](https://aws.amazon.com/marketplace/pp/prodview-64qkuwrqi4lhi) | OCR and layout analysis of document images (standard edition) |
| [YomiToku-Pro Lite - Document Analyzer](https://aws.amazon.com/marketplace/pp/prodview-zh2ubewvb6hxe) | A lightweight edition of Document Analyzer (see the [technical article](https://mlism.com/blog/tech/yomitoku-pro-lite-gpu-marketplace), in Japanese, for accuracy and processing time compared with the standard edition) |
| [YomiToku-Pro - Table Semantic Parser](https://aws.amazon.com/marketplace/pp/prodview-mqprrj24bp4w6) | Extraction of table structures (cells, key-value items and grids) from forms |

You can also find the product list in the [AWS Marketplace search results](https://aws.amazon.com/marketplace/search/results?searchTerms=yomitoku).

For deployment and analysis procedures, see the guides for the dedicated client library **YomiToku-Client**.

- **YomiToku-Client README (per-product setup and quick start)**  
  <https://github.com/MLism-Inc/yomitoku-client/blob/main/README.en.md>
- **YomiToku-Client documentation (detailed usage guide)**  
  <https://mlism-inc.github.io/yomitoku-client/>
