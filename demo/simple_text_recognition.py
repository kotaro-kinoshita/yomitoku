import cv2

from yomitoku import TextRecognizer

text_recognizer = TextRecognizer(visualize=False, device="cuda")

img = cv2.imread("demo/sample_text.jpg")
results, _ = text_recognizer(img)

for word in results.contents:
    print("Prediction Word:", word)
