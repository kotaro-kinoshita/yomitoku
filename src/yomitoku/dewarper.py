import cv2
import numpy as np
import torch

from .base import BaseModelCatalog, BaseModule
from .configs import DewarperUVDocV1Config
from .models import DewarpNet
from .schemas.document_analyzer import DewarpSchema
from .utils.dewarp import (
    draw_dewarp,
    reject_checks,
    smooth_grid,
    snap_grid,
    unwarp,
)


class DewarperModelCatalog(BaseModelCatalog):
    def __init__(self):
        super().__init__()
        self.register("uvdoc_v1", DewarperUVDocV1Config, DewarpNet)


class Dewarper(BaseModule):
    """Geometric correction of document photos: removes page curl, curvature, tilt and keystone so that text lines
    and layout become straight.

    The network predicts, for each node of a regular grid on the flat page, its position in the photo, plus a paper
    mask. The grid is smoothed, its outline snapped to the mask, and the flat page is sampled from the photo through
    it. When the prediction looks broken (see `Reject` in the config) the page is not dewarped: the input image is
    returned unchanged with status "rejected".

    `__call__` returns (DewarpSchema, image); after an accepted dewarp `last_result` holds the sampling maps used to
    map results on the dewarped image back to the input image (`utils.dewarp.reproject_results`); `visualize(img)` draws the paper mask and the grid of the last call.
    """

    model_catalog = DewarperModelCatalog()

    def __init__(
        self,
        model_name="uvdoc_v1",
        path_cfg=None,
        device="cuda",
        from_pretrained=True,
    ):
        super().__init__()
        self.load_model(
            model_name,
            path_cfg,
            from_pretrained=from_pretrained,
        )
        self.device = device
        self.model.eval()

        if self.model is not None:
            self.model.to(self.device)

        self.last_result = None
        self.last_dewarped_img = None
        self.last_grid = self.last_sdf = None
        self.last_reasons = []

    def visualize(self, img):
        """Paper mask and predicted grid of the last call drawn on its input image `img`."""
        if self.last_grid is None:
            return img.copy()
        status = "rejected" if self.last_reasons else "ok"
        return draw_dewarp(
            img, self.last_grid, self.last_sdf, status, self.last_reasons
        )

    def preprocess(self, img):
        h, w = self._cfg.data.input_size
        x = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)[:, :, ::-1]
        x = np.ascontiguousarray(x).astype(np.float32) / 255.0
        return torch.from_numpy(x).permute(2, 0, 1)[None]

    def postprocess(self, outputs):
        pp = self._cfg.postprocess
        g = outputs[0][0].float().cpu().numpy()
        sdf = outputs[2][0, 0].float().cpu().numpy() if len(outputs) > 2 else None
        if len(pp.smooth_ctrl) == 2:
            g = smooth_grid(g, tuple(pp.smooth_ctrl))
        if pp.snap and sdf is not None:
            g = snap_grid(
                g,
                sdf,
                tuple(self._cfg.data.input_size),
                max_d=pp.snap_max_dist,
                bias=pp.snap_bias,
                side_ctrl=pp.snap_side_ctrl,
                decay=pp.snap_decay,
            )
        return g, sdf

    def check(self, g, sdf, hw):
        """(failed check names, signals) for the grid of a photo of size hw = (H, W)."""
        r = self._cfg.reject
        s = reject_checks(g, sdf, hw, tuple(self._cfg.data.input_size))
        reasons = []
        if s["mask_iou"] is not None and s["mask_iou"] < r.min_mask_iou:
            reasons.append("mask_iou")
        if s["flipped"] > r.max_flipped_cells:
            reasons.append("flipped_cells")
        if s["spacing"] > r.max_spacing_ratio:
            reasons.append("spacing")
        if s["outside"] > r.max_outside:
            reasons.append("outside")
        return reasons, s

    def sample(self, img, g, sdf=None):
        """Flat image and DewarpResult for the grid g (2, nv, nu) in [-1, 1]; with the paper mask `sdf`, what lies
        outside the paper or the photo is painted with the paper colour."""
        pp = self._cfg.postprocess
        return unwarp(
            img,
            g,
            sdf,
            scale=pp.scale,
            focal_diag=pp.focal_diag,
            fill=pp.fill_background,
            fill_margin=pp.fill_margin,
        )

    def __call__(self, img, coordinate="dewarped"):
        tensor = self.preprocess(img)

        with torch.no_grad():
            tensor = tensor.to(self.device)
            with torch.autocast(
                tensor.device.type,
                dtype=torch.bfloat16,
                enabled=tensor.device.type == "cuda",
            ):
                outputs = self.model(tensor)

        g, sdf = self.postprocess(outputs)
        reasons, s = self.check(g, sdf, img.shape[:2])
        self.last_grid, self.last_sdf, self.last_reasons = g, sdf, reasons

        self.last_result = None
        self.last_dewarped_img = None
        size = None
        corrected_img = img
        if not reasons:
            corrected_img, self.last_result = self.sample(img, g, sdf)
            self.last_dewarped_img = corrected_img
            size = [corrected_img.shape[1], corrected_img.shape[0]]

        results = DewarpSchema(
            status="rejected" if reasons else "ok",
            reasons=reasons,
            coordinate=coordinate if not reasons else "photo",
            size=size,
            mask_iou=s["mask_iou"],
            flipped_cells=s["flipped"],
            spacing_ratio=s["spacing"],
            outside=s["outside"],
        )
        return results, corrected_img
