from yomitoku import DocumentAnalyzer

if __name__ == "__main__":
    configs = {"ocr": {"text_detector": {"path_cfg": "demo/text_detector.yaml"}}}

    analyzer = DocumentAnalyzer(configs=configs, visualize=True, device="cuda")
