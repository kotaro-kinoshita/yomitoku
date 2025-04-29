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
    def __init__(self, cfg, imgs, batch_quads):
        self.cfg = cfg
        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(0.5, 0.5),
            ]
        )

        self.tensors = []
        for index, (img, quads) in enumerate(zip(imgs, batch_quads)):
            validate_quads(img, quads)
            with ThreadPoolExecutor(max_workers=8) as executor:
                self.img = img
                data = list(executor.map(self.preprocess, quads))
                for tensor in data:
                    if tensor is not None:
                        tensor["img_idx"] = index
                        self.tensors.append(tensor)

    def __len__(self):
        return len(self.tensors)

    def preprocess(self, quad):
        roi_img = extract_roi_with_perspective(self.img, quad)

        if roi_img is None:
            return None

        roi_img, direction = rotate_text_image(roi_img, thresh_aspect=2)
        resized = resize_with_padding(roi_img, self.cfg.data.img_size)
        tensor = self.transform(resized)

        data = {
            "direction": direction,
            "tensor": tensor,
        }
        return data

    def __getitem__(self, index):
        # polygon = self.quads[index]
        # roi_img = extract_roi_with_perspective(self.img, polygon)
        # if roi_img is None:
        #    return
        #
        # roi_img = rotate_text_image(roi_img, thresh_aspect=2)
        # resized = resize_with_padding(roi_img, self.cfg.data.img_size)
        # tensor = self.transform(resized)

        return self.tensors[index]
