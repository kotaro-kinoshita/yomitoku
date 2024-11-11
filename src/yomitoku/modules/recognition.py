import torch

from ..data.dataset import ParseqDataset
from ..utils.misc import load_charset
from ..utils.visualizer import rec_visualizer
from . import BaseModule
from .models import PARSeq, Tokenizer


class Recognition(BaseModule):
    def __init__(self, cfg, device="cpu", visualize=False):
        super().__init__(cfg)
        self.cfg = cfg.RECOGNITION
        self.charset = load_charset(self.cfg.CHARSET)
        self.tokenizer = Tokenizer(self.charset)

        self.model = PARSeq(
            num_tokens=len(self.tokenizer),
            max_label_length=self.cfg.MODEL.MAX_LEN,
            img_size=self.cfg.DATA.IMAGE_SIZE,
            patch_size=self.cfg.MODEL.PATCH_SIZE,
            embed_dim=self.cfg.MODEL.HIDDEN_DIM,
            enc_num_heads=self.cfg.MODEL.ENC_NUM_HEADS,
            enc_mlp_ratio=self.cfg.MODEL.ENC_MLP_RATIO,
            enc_depth=self.cfg.MODEL.ENC_DEPTH,
            dec_num_heads=self.cfg.MODEL.DEC_NUM_HEADS,
            dec_mlp_ratio=self.cfg.MODEL.DEC_MLP_RATIO,
            dec_depth=self.cfg.MODEL.DEC_DEPTH,
            decode_ar=self.cfg.MODEL.DECODE_AR,
            refine_iters=self.cfg.MODEL.REFINE_ITERS,
            dropout=self.cfg.MODEL.DROPOUT,
        )

        self._device = device

        if self.cfg.WEIGHTS:
            self.model.load_state_dict(
                torch.load(self.cfg.WEIGHTS, map_location=self._device)[
                    "model"
                ]
            )

        self.model.eval()
        self.model.to(self._device)

        self.visualize = visualize

    def preprocess(self, img, polygons):
        dataset = ParseqDataset(self.cfg, img, polygons)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg.DATA.BATCH_SIZE,
            shuffle=False,
            num_workers=self.cfg.DATA.NUM_WORKERS,
        )

        return dataloader

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
                pred, score = self.tokenizer.decode(p)
                preds.extend(pred)
                scores.extend(score)

        outputs = {"contents": preds, "scores": scores, "quads": quads}

        if self.visualize:
            if vis is None:
                vis = img.copy()
            vis = rec_visualizer(
                vis,
                outputs,
                font_size=self.cfg.VISUALIZE.FONT_SIZE,
                font_color=tuple(self.cfg.VISUALIZE.COLOR[::-1]),
                font_path=self.cfg.VISUALIZE.FONT,
            )

        return outputs, vis
