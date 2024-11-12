import torch
from omegaconf import OmegaConf

from .base import BaseModule
from .configs import TextRecognizerConfig
from .data.dataset import ParseqDataset
from .models import PARSeq
from .postprocessor import ParseqTokenizer as Tokenizer
from .utils.misc import load_charset, load_config
from .utils.visualizer import rec_visualizer


class TextRecognizer(BaseModule):
    def __init__(self, cfg=None, device="cpu", visualize=False):
        super().__init__()
        self.cfg = self.set_config(cfg)
        self.charset = load_charset(self.cfg.charset)
        self.tokenizer = Tokenizer(self.charset)

        self.model = PARSeq.from_pretrained(
            self.cfg.hf_hub_repo,
            num_tokens=len(self.tokenizer),
            img_size=self.cfg.data.img_size,
            **self.cfg.parseq,
        )

        self._device = device

        self.model.eval()
        self.model.to(self._device)

        self.visualize = visualize

    def set_config(self, path_cfg):
        cfg = OmegaConf.structured(TextRecognizerConfig)
        if path_cfg is not None:
            yaml_config = load_config(path_cfg)
            cfg = OmegaConf.merge(cfg, yaml_config)
        return cfg

    def preprocess(self, img, polygons):
        dataset = ParseqDataset(self.cfg, img, polygons)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
        )

        return dataloader

    def postprocess(self, p):
        return self.tokenizer.decode(p)

    def __call__(self, img, quads, vis=None):
        """
        Apply the recognition model to the input image.

        Args:
            img (np.ndarray): target image(BGR)
            quads (list): list of quadrilaterals. Each quadrilateral is represented as a list of 4 points sorted clockwise.
            vis (np.ndarray, optional): rendering image. Defaults to None.

        Returns:
            list: list of predicted texts
            list: list of confidence scores
            np.ndarray: rendering image
        """

        dataloader = self.preprocess(img, quads)
        preds = []
        scores = []
        for data in dataloader:
            data = data.to(self._device)
            with torch.inference_mode():
                p = self.model(self.tokenizer, data).softmax(-1)
                pred, score = self.postprocess(p)
                preds.extend(pred)
                scores.extend(score)

        outputs = {"contents": preds, "scores": scores, "quads": quads}

        if self.visualize:
            if vis is None:
                vis = img.copy()
            vis = rec_visualizer(
                vis,
                outputs,
                font_size=self.cfg.visualize.font_size,
                font_color=tuple(self.cfg.visualize.color[::-1]),
                font_path=self.cfg.visualize.font,
            )

        return outputs, vis
