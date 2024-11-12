import torch

from .base import BaseModule
from .data.dataset import ParseqDataset
from .models import PARSeq
from .postprocessor import ParseqTokenizer as Tokenizer
from .utils.misc import load_charset
from .utils.visualizer import rec_visualizer


class TextRecognizer(BaseModule):
    def __init__(self, path_cfg=None, device="cuda", visualize=False):
        super().__init__()
        self.set_config(path_cfg)
        self.charset = load_charset(self._cfg.charset)
        self.tokenizer = Tokenizer(self.charset)

        self.model = PARSeq.from_pretrained(
            self._cfg.hf_hub_repo,
            cfg=self._cfg,
        )

        self.device = device

        self.model.eval()
        self.model.to(self.device)

        self.visualize = visualize

    def preprocess(self, img, polygons):
        dataset = ParseqDataset(self._cfg, img, polygons)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self._cfg.data.batch_size,
            shuffle=False,
            num_workers=self._cfg.data.num_workers,
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
            data = data.to(self.device)
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
                font_size=self._cfg.visualize.font_size,
                font_color=tuple(self._cfg.visualize.color[::-1]),
                font_path=self._cfg.visualize.font,
            )

        return outputs, vis
