from torch.utils.data import Dataset
from torchvision import transforms as T

from .functions import extract_roi_with_perspective, resize_with_padding


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

    def __len__(self):
        return len(self.quads)

    def __getitem__(self, index):
        polygon = self.quads[index]
        roi_img = extract_roi_with_perspective(self.img, polygon)
        resized = resize_with_padding(roi_img, self.cfg.data.img_size)
        tensor = self.transform(resized)

        return tensor
