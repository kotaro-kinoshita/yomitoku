from torch.utils.data import Dataset
from torchvision import transforms as T

from .functions import (
    extract_roi_with_perspective,
    resize_with_padding,
    rotate_text_image,
    validate_quads,
)

from concurrent.futures import ThreadPoolExecutor


class ParseqDataset(Dataset):
    def __init__(self, cfg, img, quads, num_workers=8):
        self.img = img[:, :, ::-1]
        self.quads = quads
        self.cfg = cfg
        self.img = img
        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(0.5, 0.5),
            ]
        )

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            data = list(executor.map(self.preprocess, self.quads))

        self.data = [tensor for tensor in data if tensor is not None]

    def preprocess(self, quad):
        if validate_quads(self.img, quad) is None:
            return None

        roi_img = extract_roi_with_perspective(self.img, quad)

        import cv2

        cv2.imwrite("sample_text.jpg", roi_img)

        if roi_img is None:
            return None

        roi_img = rotate_text_image(roi_img, thresh_aspect=2)
        resized = resize_with_padding(roi_img, self.cfg.data.img_size)

        return resized

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.transform(self.data[index])
