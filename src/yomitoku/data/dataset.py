from torch.utils.data import Dataset
from torchvision import transforms as T

from .functions import (
    extract_roi_with_perspective,
    resize_with_padding,
    rotate_text_image,
    validate_quads,
)


class ParseqDataset(Dataset):
    def __init__(self, cfg, img, quads):
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

        validate_quads(self.img, self.quads)

    def __len__(self):
        return len(self.quads)

    def __getitem__(self, index):
        polygon = self.quads[index]
        roi_img = extract_roi_with_perspective(self.img, polygon)
        if roi_img is None:
            return

        roi_img = rotate_text_image(roi_img, thresh_aspect=2)
        resized = resize_with_padding(roi_img, self.cfg.data.img_size)
        tensor = self.transform(resized)

        return tensor
