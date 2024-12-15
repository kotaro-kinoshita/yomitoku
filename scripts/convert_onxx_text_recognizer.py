import torch
from yomitoku.text_recognizer import TextRecognizer

module = TextRecognizer(device="cpu")
model = module.model
img_size = module._cfg.data.img_size
input = torch.randn(1, 3, *img_size, requires_grad=True)
dynamic_axes = {
    "input": {0: "batch_size"},
    "output": {0: "batch_size"},
}

print(input.shape)

torch.onnx.export(
    model,
    input,
    "onnx/yomitoku-text-recognizer-parseq-open-beta.onnx",
    opset_version=14,
    input_names=["input"],
    output_names=["output"],
    do_constant_folding=True,
    dynamic_axes=dynamic_axes,
)
