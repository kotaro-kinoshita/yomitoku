import cv2
import numpy as np
import torch

from .base import BaseModelCatalog, BaseModule
from .configs import DewarperPageNetV1Config, DewarperPageNetV2Config
from .models import PageNet
from .models.pagenet import decode, grid_corners
from .schemas.document_analyzer import DewarpSchema
from .utils.dewarp import (
    content_angle,
    edge_angle,
    fill_background,
    grid_mask,
    grid_stats,
    plane_frontal,
    reject_checks,
    resample_grid,
    rigid_grid,
    rotate_crop,
    rotate_grid,
    seen_box,
    snap_grid,
    unwarp,
)

_COLORS = [
    (0, 0, 230),
    (230, 120, 0),
    (0, 170, 0),
    (200, 0, 200),
    (0, 200, 230),
    (120, 60, 0),
    (0, 120, 255),
    (160, 160, 0),
]


class PageDewarperModelCatalog(BaseModelCatalog):
    def __init__(self):
        super().__init__()
        self.register("pagenet_v1", DewarperPageNetV1Config, PageNet)
        self.register("pagenet_v2", DewarperPageNetV2Config, PageNet)


class PageDewarper(BaseModule):
    """Geometric correction of every page in a photo: the two pages of a spread, receipts or business cards side by
    side, ... are found and each is dewarped on its own (page curl, curvature, tilt and keystone removed).

    The network detects the pages and predicts, for each, a regular grid on the flat page with its position in the
    photo and whether each node is seen. Detected pages are filtered (see `Filter` in the config): collapsed grids,
    pages lying inside a larger page, and pages cut off by the frame (except the best page). Each page kept is cut to
    the bounding box of its seen part and sampled from the photo; what lies outside the photo (and, with the paper
    SDF head, off the paper) is painted with the paper colour. When no page is found the input image is returned
    unchanged with status "rejected".

    pagenet_v2 (default) adds the paper SDF head: each page's outline is snapped to the paper edge (`Snap` in the
    config). Pages that the 3D head finds flat and facing the camera (PDF renders, scans; `Flat`) are not unwarped
    through their grid: the photo is only rotated (one affine warp) so that the page is level, the angle refined from
    the page's text lines or paper edges, and cropped to the page (DewarpSchema `flat`, `rotation`).

    `__call__` returns a list of (DewarpSchema, image), one per page, left to right then top to bottom; after the
    call `last_results` holds the sampling maps of each page (`utils.dewarp.reproject_results`) and `visualize(img)`
    draws the pages found. `last_result` / `last_dewarped_img` are those of the first page, as in `Dewarper`.
    """

    model_catalog = PageDewarperModelCatalog()

    def __init__(
        self,
        model_name="pagenet_v2",
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

        # paper SDF of the last photo (network input / 4), + on any page (SDF models)
        self.last_sdf = None
        self.last_pages = []
        self.last_results = []
        self.last_dewarped_imgs = []
        self.last_result = None
        self.last_dewarped_img = None

    def preprocess(self, img):
        h, w = self._cfg.data.input_size
        x = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)[:, :, ::-1]
        x = np.ascontiguousarray(x).astype(np.float32) / 255.0
        return torch.from_numpy(x).permute(2, 0, 1)[None]

    @torch.no_grad()
    def detect(self, img):
        """Pages of the photo before filtering: [{score, grid (2, nv, nu) [-1, 1], grid3d (3, nv, nu), seen (nv, nu)
        probability, corners}], best first. Sets `last_sdf` (SDF models)."""
        det = self._cfg.detect
        x = self.preprocess(img).to(self.device)
        with torch.autocast(
            x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"
        ):
            heat, off, feats = self.model(x)
            sdf = self.model.sdf_map(feats)
            self.last_sdf = None if sdf is None else sdf[0, 0].float().cpu().numpy()
            dets = decode(
                heat,
                off,
                tuple(self._cfg.data.input_size),
                thresh=det.thresh,
                k=det.max_pages,
            )[0]
            if not dets:
                return []
            corners = torch.stack([c for _, c in dets])
            idx = torch.zeros(len(dets), dtype=torch.long, device=x.device)
            for _ in range(det.refine + 1):
                g2, g3, seen = self.model.roi(feats, x, idx, corners)
                corners = grid_corners(g2.float())
        pages = []
        for (s, _), g, q, sn, c in zip(
            dets,
            g2.float().cpu().numpy(),
            g3.float().cpu().numpy(),
            torch.sigmoid(seen.float())[:, 0].cpu().numpy(),
            corners.cpu().numpy(),
        ):
            if any(np.abs(c - p["corners"]).sum(-1).mean() < det.dedup for p in pages):
                continue
            pages.append({"score": s, "grid": g, "grid3d": q, "seen": sn, "corners": c})
        return pages

    def refine(self, img, pages):
        """Per page, in place: flat pages (`Flat` in the config) get the rectangle fitted to their grid, turned by the
        deskew angle ("flat" True, "deskew_deg", "deskew_src": content / edge / none); the others, with the SDF head,
        get their outline snapped to the paper edge ("snapped"). "grid_raw" keeps the predicted grid."""
        fl, sn = getattr(self._cfg, "flat", None), getattr(self._cfg, "snap", None)
        for p in pages:
            g = p["grid"]
            p["flat"] = False
            if fl is not None and fl.enabled and "grid3d" in p:
                flatness, tilt = plane_frontal(p["grid3d"])
                p["flat"] = flatness < fl.max_flatness and tilt < fl.max_tilt
            if p["flat"]:
                hw = img.shape[:2]
                p["grid_raw"], r = g, rigid_grid(g, hw)
                a, src = 0.0, "none"
                if fl.deskew:
                    # text lines / rules first (on scans the print can sit askew on the sheet); paper edges on
                    # (nearly) blank pages
                    c = content_angle(img, r, max_deg=fl.deskew_max_deg)
                    if c is not None:
                        a, src = c, "content"
                    else:
                        e = edge_angle(img, r, max_deg=fl.edge_max_deg)
                        if e is not None:
                            a, src = e, "edge"
                p["deskew_deg"], p["deskew_src"] = a, src
                p["grid"] = rotate_grid(r, a, hw) if a else r
            elif sn is not None and sn.enabled and self.last_sdf is not None:
                gs = snap_grid(
                    g,
                    self.last_sdf,
                    tuple(self._cfg.data.input_size),
                    max_d=sn.max_dist,
                    bias=sn.bias,
                    side_ctrl=sn.side_ctrl,
                    decay=sn.decay,
                )
                r, n = grid_stats(g), grid_stats(gs)
                ok = n["flip"] <= r["flip"] and n["cell_min"] >= 0.8 * r["cell_min"]
                p["grid_raw"], p["grid"], p["snapped"] = g, gs if ok else g, ok
        return pages

    def filter(self, pages):
        """Drop collapsed, nested and cut-off pages (in this order); adds "stats" to each page kept."""
        f = self._cfg.filter
        for p in pages:
            p["stats"] = {
                **grid_stats(p["grid"]),
                "seen": float((p["seen"] > 0.5).mean()),
            }
        pages = [
            p
            for p in pages
            if p["stats"]["flip"] == 0 and p["stats"]["cell_min"] >= f.min_cell
        ]
        masks = [grid_mask(p["grid"]) for p in pages]
        areas = [m.sum() for m in masks]
        pages = [
            p
            for i, p in enumerate(pages)
            if not any(
                areas[j] > areas[i]
                and (masks[i] & masks[j]).sum() >= f.nest * max(areas[i], 1)
                for j in range(len(pages))
                if j != i
            )
        ]
        return [
            p
            for i, p in enumerate(pages)
            if i == 0
            or (
                p["stats"]["inside"] >= f.min_inside
                and p["stats"]["seen"] >= f.min_seen
            )
        ]

    def pages(self, img):
        """Pages kept, in reading position: left to right, then top to bottom (by the centre of the seen part)."""
        pages = self.filter(self.refine(img, self.detect(img)))

        def centre(p):
            g = p["grid"]
            m = p["seen"] > 0.5
            pts = g[:, m] if m.any() else g.reshape(2, -1)
            return pts.mean(1)

        rows = sorted(pages, key=lambda p: centre(p)[1])
        # pages whose centres differ in y by less than half a page height are on one row
        ordered, row = [], []
        for p in rows:
            if row and centre(p)[1] - centre(row[0])[1] > 0.5 * _height(row[0]):
                ordered += sorted(row, key=lambda q: centre(q)[0])
                row = []
            row.append(p)
        return ordered + sorted(row, key=lambda q: centre(q)[0])

    def visualize(self, img):
        """Pages of the last call drawn on its input image `img`: whole grid thin, the part dewarped thick, with the
        page index and detection score."""
        out = img.copy()
        H, W = img.shape[:2]
        t = max(1, round(max(H, W) / 700))
        for k, p in enumerate(self.last_pages):
            c = _COLORS[k % len(_COLORS)]
            P = (p["grid"].transpose(1, 2, 0) + 1) / 2 * np.array([W, H], np.float32)
            for j in range(0, P.shape[0], 2):
                cv2.polylines(
                    out, [np.round(P[j]).astype(np.int32)], False, c, 1, cv2.LINE_AA
                )
            for i in range(0, P.shape[1], 2):
                cv2.polylines(
                    out, [np.round(P[:, i]).astype(np.int32)], False, c, 1, cv2.LINE_AA
                )
            Q = (p["cut_grid"].transpose(1, 2, 0) + 1) / 2 * np.array([W, H])
            ring = np.concatenate([Q[0], Q[1:, -1], Q[-1, ::-1][1:], Q[::-1, 0][1:]])
            cv2.polylines(
                out, [np.round(ring).astype(np.int32)], True, c, 3 * t, cv2.LINE_AA
            )
            at = tuple(int(v) for v in np.round(Q.mean((0, 1))))
            label = f"{k} {p['score']:.2f}"
            if p.get("flat"):
                label += f" flat {p['rotation']:+.2f}"
            for col, wd in (((255, 255, 255), 5 * t), (c, 2 * t)):
                cv2.putText(
                    out,
                    label,
                    at,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9 * t,
                    col,
                    wd,
                    cv2.LINE_AA,
                )
        return out

    def __call__(self, img, coordinate="dewarped"):
        pp = self._cfg.postprocess
        H, W = img.shape[:2]
        pages = self.pages(img)
        self.last_pages, self.last_results, self.last_dewarped_imgs = [], [], []
        self.last_result = self.last_dewarped_img = None

        if not pages:
            results = DewarpSchema(
                status="rejected",
                reasons=["no_page"],
                coordinate="photo",
                size=None,
                mask_iou=None,
                flipped_cells=0.0,
                spacing_ratio=0.0,
                outside=0.0,
                num_pages=0,
            )
            return [(results, img)]

        outputs = []
        for k, p in enumerate(pages):
            g = p["grid"]
            if p.get("flat"):
                # no grid unwarp: the photo rotated level and cropped to the page
                out, res, M = rotate_crop(img, g, scale=self._cfg.flat.scale)
                if pp.fill_background:
                    fill_background(out, res.paper)
                p["affine"] = M  # 2x3, photo px -> output px
                p["rotation"] = float(np.degrees(np.arctan2(M[0, 1], M[0, 0])))
                g = res.grid
            else:
                if pp.cut_seen:
                    lo, hi = seen_box(p["seen"])
                    g = resample_grid(g, lo, hi)
                out, res = unwarp(
                    img,
                    g,
                    sdf=self.last_sdf,
                    scale=pp.scale,
                    focal_diag=pp.focal_diag,
                    fill=pp.fill_background,
                )
            p["cut_grid"] = g
            s = reject_checks(g, None, (H, W), tuple(self._cfg.data.input_size))
            schema = DewarpSchema(
                status="ok",
                reasons=[],
                coordinate=coordinate,
                size=[out.shape[1], out.shape[0]],
                mask_iou=None,
                flipped_cells=s["flipped"],
                spacing_ratio=s["spacing"],
                outside=s["outside"],
                page=k,
                num_pages=len(pages),
                score=float(p["score"]),
                flat=True if p.get("flat") else None,
                rotation=round(p["rotation"], 3) if p.get("flat") else None,
            )
            self.last_pages.append(p)
            self.last_results.append(res)
            self.last_dewarped_imgs.append(out)
            outputs.append((schema, out))

        self.last_result = self.last_results[0]
        self.last_dewarped_img = self.last_dewarped_imgs[0]
        return outputs


def _height(p):
    """Height of a page's seen part, in normalized photo coordinates."""
    g = p["grid"]
    m = p["seen"] > 0.5
    y = g[1][m] if m.any() else g[1].ravel()
    return float(y.max() - y.min())
