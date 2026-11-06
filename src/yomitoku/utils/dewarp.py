"""Geometric correction (dewarping) helpers, ported from docwarp-synth (`docwarp_synth/dewarp/infer.py`, numpy
instead of scipy): grid post-processing (B-spline smoothing, snapping the outline to the paper mask), the output
aspect, the sampling maps, checks that reject a broken dewarp, and the mapping of results on the dewarped image back
to the photo (rectangles become curved polygons)."""

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional
from pydantic import BaseModel


def _bspline_basis(n: int, k: int) -> np.ndarray:
    """(n, k) clamped cubic B-spline basis with k control points, evaluated at n uniform parameters in [0, 1]."""
    t = np.concatenate([[0.0] * 3, np.linspace(0, 1, k - 2), [1.0] * 3])
    x = np.linspace(0, 1, n)
    B = np.zeros((n, len(t) - 1))
    for i in range(len(t) - 1):  # degree 0 (the last knot interval is closed)
        B[:, i] = (t[i] <= x) & (
            (x < t[i + 1]) | ((x == 1.0) & (t[i + 1] == 1.0) & (t[i] < 1.0))
        )
    for d in range(1, 4):  # Cox-de Boor recursion
        Bn = np.zeros((n, len(t) - 1 - d))
        for i in range(len(t) - 1 - d):
            a = (x - t[i]) / (t[i + d] - t[i]) if t[i + d] > t[i] else 0.0
            b = (
                (t[i + d + 1] - x) / (t[i + d + 1] - t[i + 1])
                if t[i + d + 1] > t[i + 1]
                else 0.0
            )
            Bn[:, i] = a * B[:, i] + b * B[:, i + 1]
        B = Bn
    return B


def smooth_grid(g: np.ndarray, ctrl: tuple[int, int]) -> np.ndarray:
    """Least-squares fit of a (2, nv, nu) grid by a tensor cubic B-spline surface with ctrl = (cu, cv) control
    points: removes node-level jitter, keeps shape variation coarser than ~1 / ctrl of the page."""
    nv, nu = g.shape[1:]
    Bu, Bv = _bspline_basis(nu, ctrl[0]), _bspline_basis(nv, ctrl[1])
    Pu, Pv = Bu @ np.linalg.pinv(Bu), Bv @ np.linalg.pinv(Bv)
    return np.einsum("ij,cjk,lk->cil", Pv, g, Pu).astype(g.dtype)


def _outer_ring(nv: int, nu: int) -> list[tuple[int, int, int, int]]:
    """Outer nodes (j, i) with the inward neighbour (jn, in_) that defines their outward direction."""
    ring = [(0, i, 1, min(max(i, 1), nu - 2)) for i in range(nu)] + [
        (nv - 1, i, nv - 2, min(max(i, 1), nu - 2)) for i in range(nu)
    ]
    ring += [(j, 0, min(max(j, 1), nv - 2), 1) for j in range(1, nv - 1)] + [
        (j, nu - 1, min(max(j, 1), nv - 2), nu - 2) for j in range(1, nv - 1)
    ]
    return ring


def _bilinear(img: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Bilinear samples of a 2D array at fractional (y, x), clamped at the border."""
    h, w = img.shape
    y, x = np.clip(y, 0, h - 1), np.clip(x, 0, w - 1)
    y0, x0 = (
        np.minimum(np.floor(y).astype(int), h - 2),
        np.minimum(np.floor(x).astype(int), w - 2),
    )
    fy, fx = y - y0, x - x0
    return (
        img[y0, x0] * (1 - fy) * (1 - fx)
        + img[y0, x0 + 1] * (1 - fy) * fx
        + img[y0 + 1, x0] * fy * (1 - fx)
        + img[y0 + 1, x0 + 1] * fy * fx
    )


def sdf_crossings(
    g: np.ndarray, sdf: np.ndarray, input_hw=(712, 488), max_d: float = 6.0
):
    """For every outer node of the grid: its outward direction (input px) and the signed distance t along it to the
    nearest SDF = 0 crossing within +-max_d px (NaN when there is none or the search leaves the image)."""
    H, W = input_hw
    nv, nu = g.shape[1:]
    P = (g.transpose(1, 2, 0) + 1) / 2 * np.array([W, H])
    s_t = np.linspace(-max_d, max_d, int(8 * max_d) + 1)
    N, T = np.zeros((nv, nu, 2)), np.full((nv, nu), np.nan)
    for j, i, jn, in_ in _outer_ring(nv, nu):
        n = P[j, i] - P[jn, in_]
        n /= max(np.linalg.norm(n), 1e-6)
        N[j, i] = n
        q = P[j, i] + s_t[:, None] * n
        if (q < 0).any() or (q[:, 0] > W).any() or (q[:, 1] > H).any():
            continue
        s = _bilinear(sdf, q[:, 1] / 4 - 0.5, q[:, 0] / 4 - 0.5)
        k = np.nonzero((s[:-1] > 0) & (s[1:] <= 0))[
            0
        ]  # inside -> outside, going outward
        if len(k):
            k = k[np.argmin(np.abs(s_t[k]))]
            T[j, i] = s_t[k] + (s_t[k + 1] - s_t[k]) * s[k] / (s[k] - s[k + 1])
    return N, T


def _smooth_curve(Q: np.ndarray, k: int) -> np.ndarray:
    m = np.isfinite(Q).all(1)
    if m.sum() < 2 * k:
        return Q
    B = _bspline_basis(len(Q), k)
    return np.where(m[:, None], B @ np.linalg.lstsq(B[m], Q[m], rcond=None)[0], np.nan)


def snap_grid(
    g: np.ndarray,
    sdf: np.ndarray,
    input_hw=(712, 488),
    max_d: float = 10.0,
    bias: float = -0.06,
    side_ctrl: int = 12,
    decay: float = 3.0,
) -> np.ndarray:
    """Move the grid outline to the SDF head's zero level minus `bias` px (sides smoothed with `side_ctrl` control
    points) and carry the displacement inward over a band of ~`decay` nodes."""
    H, W = input_hw
    nv, nu = g.shape[1:]
    N, T = sdf_crossings(g, sdf, input_hw, max_d)
    P0 = (g.transpose(1, 2, 0) + 1) / 2 * np.array([W, H])
    Q = P0 + (T - bias)[..., None] * N
    if side_ctrl:
        for sl in (np.s_[0, :], np.s_[-1, :], np.s_[:, 0], np.s_[:, -1]):
            Qs = _smooth_curve(Q[sl], side_ctrl)
            if sl[0] == slice(None):
                Q[1:-1, sl[1]] = Qs[1:-1]
            else:
                Q[sl] = Qs
    D = np.nan_to_num(((Q - P0) * N).sum(-1))[..., None] * N
    fu, fv = np.exp(-np.arange(nu) / decay), np.exp(-np.arange(nv) / decay)
    wl, wr, wt, wb = (
        fu[None, :, None],
        fu[::-1][None, :, None],
        fv[:, None, None],
        fv[::-1][:, None, None],
    )
    Tp, Bt, L, R = D[0][None], D[-1][None], D[:, 0][:, None], D[:, -1][:, None]
    C = (
        wl * wt * D[0, 0]
        + wr * wt * D[0, -1]
        + wl * wb * D[-1, 0]
        + wr * wb * D[-1, -1]
    )
    P = P0 + wt * Tp + wb * Bt + wl * L + wr * R - C
    return (P / np.array([W, H]) * 2 - 1).transpose(2, 0, 1).astype(g.dtype)


def _arc(g: np.ndarray, axis: int) -> float:
    """Mean polyline length of grid rows (axis=1) or columns (axis=0); g (nv, nu, C)."""
    return float(np.linalg.norm(np.diff(g, axis=axis), axis=-1).sum(axis).mean())


def output_aspect(
    px: np.ndarray, hw: tuple[int, int], f_diag: float = 0.6, block: int = 4
) -> float:
    """Width / height of the flat output from the grid in photo pixels `px` (nv, nu, 2): every `block` x `block`
    patch of grid cells is the same rectangle on the page, so its image quad and a pinhole camera (focal `f_diag` x
    photo diagonal, a phone's ~26 mm equivalent; principal point at the centre) give that rectangle's w/h; the median
    over patches inside the photo, scaled by the cell counts."""
    H, W = hw
    nv, nu = px.shape[:2]
    c = np.array([W / 2, H / 2])
    f = f_diag * np.hypot(W, H)
    sq = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    est, est_all = [], []
    for j in range(0, nv - block, block):
        for i in range(0, nu - block, block):
            q = np.array(
                [px[j, i], px[j, i + block], px[j + block, i + block], px[j + block, i]]
            )
            Hm = cv2.getPerspectiveTransform(sq, (q - c).astype(np.float32))
            r1, r2 = Hm[:, 0] * [1 / f, 1 / f, 1], Hm[:, 1] * [1 / f, 1 / f, 1]
            a = np.log(np.linalg.norm(r1) / max(np.linalg.norm(r2), 1e-12))
            est_all.append(a)
            if (q >= 0).all() and (q[:, 0] < W).all() and (q[:, 1] < H).all():
                est.append(a)
    return float(np.exp(np.median(est or est_all))) * (nu - 1) / (nv - 1)


@dataclass
class DewarpResult:
    grid: np.ndarray  # (2, nv, nu) in [-1, 1] photo coordinates
    map_x: np.ndarray  # (h, w) photo x of every flat pixel
    map_y: np.ndarray  # (h, w) photo y of every flat pixel
    aspect: float  # flat width / height
    paper: Optional[np.ndarray] = (
        None  # (h, w) bool, flat pixels on the paper (not desk, not outside the photo)
    )

    def to_photo(self, pts) -> np.ndarray:
        """Flat-image points (..., 2) [x, y] -> photo points (..., 2) (bilinear lookup in the sampling maps)."""
        p = np.asarray(pts, np.float64)
        flat = p.reshape(-1, 2)
        x = _bilinear(self.map_x, flat[:, 1], flat[:, 0])
        y = _bilinear(self.map_y, flat[:, 1], flat[:, 0])
        return np.stack([x, y], -1).reshape(p.shape)


def paper_mask(map_x, map_y, sdf, hw, margin=0.0) -> np.ndarray:
    """(h, w) bool: the pixels of the dewarped image that come from the paper (SDF at or above `margin`; SDF clipped
    to ±1) and from inside the photo of size hw = (H, W)."""
    H, W = hw
    keep = (map_x >= 0) & (map_x <= W - 1) & (map_y >= 0) & (map_y <= H - 1)
    if sdf is not None:
        sh, sw = sdf.shape
        sx = ((map_x + 0.5) * sw / W - 0.5).astype(np.float32)
        sy = ((map_y + 0.5) * sh / H - 0.5).astype(np.float32)
        s = cv2.remap(
            sdf.astype(np.float32),
            sx,
            sy,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        keep &= s >= margin
    return keep


def fill_background(flat, paper) -> None:
    """In place: paints the pixels of the dewarped image `flat` off the `paper` mask (desk, a neighbouring page,
    outside the photo) with the paper colour (median of the paper pixels)."""
    if paper.all() or not paper.any():
        return
    flat[~paper] = np.median(flat[paper][::17], axis=0).astype(flat.dtype)


def unwarp(img, g, sdf=None, scale=1.5, focal_diag=0.6, fill=True, fill_margin=0.0):
    """Flat image and DewarpResult for the grid g (2, nv, nu) in [-1, 1] of `img`: output aspect from the grid
    (`output_aspect`), long side `scale` x the page's size in the photo; with `fill`, what lies outside the paper
    (`sdf`, when given) or the photo is painted with the paper colour."""
    H, W = img.shape[:2]
    px = (g.transpose(1, 2, 0) + 1) / 2 * np.array([W, H], np.float32)
    aspect = output_aspect(px, (H, W), f_diag=focal_diag)
    long_px = max(_arc(px, 1), _arc(px, 0)) * scale
    long_px = int(np.clip(long_px, 256, 1.25 * scale * max(H, W)))
    out_w, out_h = (
        (long_px, max(1, round(long_px / aspect)))
        if aspect >= 1
        else (max(1, round(long_px * aspect)), long_px)
    )
    gu = F.interpolate(
        torch.from_numpy(np.ascontiguousarray(g))[None].float(),
        size=(out_h, out_w),
        mode="bicubic",
        align_corners=True,
    )[0].numpy()
    mx = (((gu[0] + 1) * W - 1) / 2).astype(np.float32)
    my = (((gu[1] + 1) * H - 1) / 2).astype(np.float32)
    flat = cv2.remap(
        img,
        mx,
        my,
        cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    paper = paper_mask(mx, my, sdf, (H, W), margin=fill_margin)
    if fill:
        fill_background(flat, paper)
    return flat, DewarpResult(g, mx, my, aspect, paper)


def drop_off_paper_figures(figures, paper, max_off=0.5, max_cover=0.8):
    """Figures of the layout analysis on a dewarped image without those that are not on the page: mostly off the
    `paper` mask (share of the box > max_off: desk or the blank band of a page cut by the photo frame), or containing
    the page itself (box covering > max_cover of the paper)."""
    h, w = paper.shape
    total = max(int(paper.sum()), 1)
    kept = []
    for f in figures:
        x1, y1, x2, y2 = (int(v) for v in f.box)
        x1, x2 = max(x1, 0), min(x2, w)
        y1, y2 = max(y1, 0), min(y2, h)
        area = max((x2 - x1) * (y2 - y1), 1)
        on = int(paper[y1:y2, x1:x2].sum())
        if 1 - on / area > max_off or on / total > max_cover:
            continue
        kept.append(f)
    return kept


def reject_checks(g: np.ndarray, sdf, hw, input_hw) -> dict:
    """Signals that the predicted grid (2, nv, nu) in [-1, 1] is broken, for a photo of size hw = (H, W):
    mask_iou     IoU of the grid outline and the paper mask (SDF > 0); None without an SDF head
    flipped      share of grid cells turned over (the page folded onto itself)
    spacing      longest / median grid edge (a grid squeezed into a strip or a corner)
    outside      share of grid nodes more than 5 % of the photo size outside it"""
    H, W = hw
    P = (g.transpose(1, 2, 0) + 1) / 2 * np.array([W, H])
    a, b, c = P[:-1, :-1], P[:-1, 1:], P[1:, :-1]
    cross = (b - a)[..., 0] * (c - a)[..., 1] - (b - a)[..., 1] * (c - a)[..., 0]
    s = np.sign(np.median(cross))
    flipped = float(np.mean(cross * s <= 0))
    e = np.concatenate(
        [
            np.linalg.norm(np.diff(P, axis=0), axis=-1).ravel(),
            np.linalg.norm(np.diff(P, axis=1), axis=-1).ravel(),
        ]
    )
    spacing = float(e.max() / max(np.median(e), 1e-6))
    m = 0.05 * np.array([W, H])
    outside = float(np.mean((P < -m).any(-1) | (P > np.array([W, H]) + m).any(-1)))
    iou = None
    if sdf is not None:
        h, w = input_hw
        q = (g.transpose(1, 2, 0) + 1) / 2 * np.array([w, h])
        poly = np.concatenate([q[0], q[1:, -1], q[-1, :-1][::-1], q[1:-1, 0][::-1]])
        gm = np.zeros((h, w), np.uint8)
        cv2.fillPoly(gm, [np.round(poly * 16).astype(np.int32)], 1, shift=4)
        sm = cv2.resize(sdf, (w, h), interpolation=cv2.INTER_LINEAR) >= 0
        gm = gm.astype(bool)
        iou = float((gm & sm).sum() / max((gm | sm).sum(), 1))
    return {"mask_iou": iou, "flipped": flipped, "spacing": spacing, "outside": outside}


N_EDGE = 8  # samples per polygon edge


def _outline(corners: np.ndarray, n: int = N_EDGE) -> np.ndarray:
    """Points along the closed outline through `corners` (k, 2), n per edge (each edge's end point excluded)."""
    t = np.linspace(0, 1, n, endpoint=False)[:, None]
    return np.concatenate(
        [a + t * (b - a) for a, b in zip(corners, np.roll(corners, -1, 0))]
    )


def _box_corners(box) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float64)


def _clip(P: np.ndarray, hw) -> np.ndarray:
    H, W = hw
    return np.stack([np.clip(P[:, 0], 0, W - 1), np.clip(P[:, 1], 0, H - 1)], 1)


def _map_box(d: dict, dw: DewarpResult, hw) -> None:
    """In place: d["box"] (flat [x1, y1, x2, y2]) -> photo bounding box, d["polygon"] -> photo outline."""
    P = _clip(dw.to_photo(_outline(_box_corners(d["box"]))), hw)
    d["polygon"] = np.round(P).astype(int).tolist()
    d["box"] = [
        int(np.floor(P[:, 0].min())),
        int(np.floor(P[:, 1].min())),
        int(np.ceil(P[:, 0].max())),
        int(np.ceil(P[:, 1].max())),
    ]


def _map_word(w: dict, dw: DewarpResult, hw) -> None:
    Q = np.asarray(w["points"], np.float64)
    w["points"] = np.round(_clip(dw.to_photo(Q), hw)).astype(int).tolist()
    w["polygon"] = np.round(_clip(dw.to_photo(_outline(Q)), hw)).astype(int).tolist()


def reproject_results(res, dw: DewarpResult, hw):
    """Map every `box` ([x1, y1, x2, y2]) and word `points` of a result schema (DocumentAnalyzer,
    TableSemanticParser, ...) from the dewarped image back to the photo `hw` = (H, W), in place: each element with a
    box gets its curved outline in `polygon` and `box` becomes the outline's bounding box; word points are mapped
    corner by corner. Returns `res`."""
    seen = set()

    def walk(o):
        if isinstance(o, BaseModel):
            if id(o) in seen:
                return
            seen.add(id(o))
            fields = type(o).model_fields
            if "points" in fields and "polygon" in fields and o.points is not None:
                d = {"points": o.points}
                _map_word(d, dw, hw)
                o.polygon, o.points = d["polygon"], d["points"]
            elif "box" in fields and "polygon" in fields and o.box is not None:
                d = {"box": o.box}
                _map_box(d, dw, hw)
                o.polygon, o.box = d["polygon"], d["box"]
            for name in fields:
                if name not in ("box", "points", "polygon"):
                    walk(getattr(o, name))
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)

    walk(res)
    return res


def draw_dewarp(img, grid, sdf, status="ok", reasons=()):
    """The dewarper's view of a photo: the paper mask (SDF > 0; outside darkened, outline yellow) and the predicted
    grid (rows red, columns blue, outer frame green), with the accept / reject status."""
    H, W = img.shape[:2]
    out = img.copy()
    t = max(1, round(max(H, W) / 800))
    if sdf is not None:
        m = cv2.resize(sdf, (W, H), interpolation=cv2.INTER_LINEAR) >= 0
        out[~m] = (out[~m] * 0.35).astype(np.uint8)
        cs, _ = cv2.findContours(
            m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        cv2.drawContours(out, cs, -1, (0, 220, 255), 2 * t, cv2.LINE_AA)
    P = np.stack([((grid[0] + 1) * W - 1) / 2, ((grid[1] + 1) * H - 1) / 2], -1)
    for j in range(P.shape[0]):
        cv2.polylines(
            out, [np.round(P[j]).astype(np.int32)], False, (0, 0, 255), t, cv2.LINE_AA
        )
    for i in range(P.shape[1]):
        cv2.polylines(
            out,
            [np.round(P[:, i]).astype(np.int32)],
            False,
            (255, 0, 0),
            t,
            cv2.LINE_AA,
        )
    ring = np.concatenate([P[0], P[1:, -1], P[-1, :-1][::-1], P[1:-1, 0][::-1]])
    cv2.polylines(
        out, [np.round(ring).astype(np.int32)], True, (0, 200, 0), 3 * t, cv2.LINE_AA
    )
    label = (
        "dewarp: ok" if status == "ok" else f"dewarp: rejected ({', '.join(reasons)})"
    )
    color = (0, 160, 0) if status == "ok" else (0, 0, 255)
    fs = max(0.6, max(H, W) / 1500)
    cv2.putText(
        out,
        label,
        (10, int(40 * fs)),
        cv2.FONT_HERSHEY_SIMPLEX,
        fs,
        (255, 255, 255),
        6 * t,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        label,
        (10, int(40 * fs)),
        cv2.FONT_HERSHEY_SIMPLEX,
        fs,
        color,
        2 * t,
        cv2.LINE_AA,
    )
    return out


_COLORS = {  # BGR, as in the box visualizations
    "word": (0, 0, 255),
    "paragraph": (255, 0, 0),
    "table": (0, 255, 0),
    "figure": (0, 0, 255),
    "cell": (255, 0, 255),
}


def _poly(img, pts, color, t):
    if pts:
        cv2.polylines(img, [np.asarray(pts, np.int32)], True, color, t, cv2.LINE_AA)


def draw_document(img, results, words_only=False):
    """Results mapped back to the photo, drawn as curved outlines."""
    out = img.copy()
    t = max(2, round(max(img.shape[:2]) / 1000))
    if words_only:
        for w in results.words:
            _poly(out, w.polygon or w.points, _COLORS["word"], max(1, t // 2))
        return out
    for tb in results.tables:
        for c in tb.cells:
            _poly(out, c.polygon, _COLORS["cell"], t)
        _poly(out, tb.polygon, _COLORS["table"], t)
    for f in results.figures:
        _poly(out, f.polygon, _COLORS["figure"], t)
    for p in results.paragraphs:
        _poly(out, p.polygon, _COLORS["paragraph"], t)
    return out


def grid_stats(g: np.ndarray) -> dict:
    """Shape checks of a page grid (2, nv, nu) in [-1, 1], for the multi-page dewarper:
    inside    share of grid nodes inside the photo
    flip      share of grid cells folded against the majority orientation
    cell_min  2nd-percentile / median cell area (a grid collapsed into a strip or a point)"""
    a, b, c = g[:, :-1, :-1], g[:, :-1, 1:], g[:, 1:, :-1]
    cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    area = np.abs(cross)
    return {
        "inside": float((np.abs(g) <= 1).all(0).mean()),
        "flip": float((np.sign(cross) != np.sign(np.median(cross))).mean()),
        "cell_min": float(np.percentile(area, 2) / max(float(np.median(area)), 1e-9)),
    }


def grid_mask(g: np.ndarray, size: int = 256) -> np.ndarray:
    """(size, size) bool mask of the photo covered by the outline of a grid (2, nv, nu) in [-1, 1]."""
    P = (g.transpose(1, 2, 0) + 1) / 2 * (size - 1)
    ring = np.concatenate([P[0], P[1:, -1], P[-1, ::-1][1:], P[::-1, 0][1:]])
    m = np.zeros((size, size), np.uint8)
    cv2.fillPoly(m, [np.round(ring).astype(np.int32)], 1)
    return m.astype(bool)


def seen_box(seen: np.ndarray, thresh: float = 0.5, up: int = 4, frac: float = 0.0):
    """uv box (lo, hi) of a page's seen part from its node `seen` probabilities (nv, nu): trim columns, then rows,
    from each side while no more than `frac` of the side line is seen. frac = 0 keeps every line with a seen node
    (the bounding box of the seen part: nothing in the photo is cut off; what lies outside it is filled)."""
    nv, nu = seen.shape
    m = (
        cv2.resize(
            seen.astype(np.float32),
            ((nu - 1) * up + 1, (nv - 1) * up + 1),
            interpolation=cv2.INTER_LINEAR,
        )
        > thresh
    )
    if m.all() or not m.any():
        return np.zeros(2, np.float32), np.ones(2, np.float32)
    cols = np.nonzero(m.mean(0) > frac)[0]
    cols = cols if len(cols) else np.array([int(np.argmax(m.mean(0)))])
    c0, c1 = cols[0], cols[-1]
    rows = np.nonzero(m[:, c0 : c1 + 1].mean(1) > frac)[0]
    rows = rows if len(rows) else np.array([int(np.argmax(m[:, c0 : c1 + 1].mean(1)))])
    W, H = m.shape[1] - 1, m.shape[0] - 1
    return (
        np.array([c0 / W, rows[0] / H], np.float32),
        np.array([c1 / W, rows[-1] / H], np.float32),
    )


def resample_grid(g: np.ndarray, lo, hi) -> np.ndarray:
    """Grid (2, nv, nu) sampled at uniform uv in [0, 1]^2 -> the same node count over the uv box [lo, hi]
    (bicubic)."""
    nv, nu = g.shape[1:]
    mx = (lo[0] + (hi[0] - lo[0]) * np.linspace(0, 1, nu, dtype=np.float32)) * (nu - 1)
    my = (lo[1] + (hi[1] - lo[1]) * np.linspace(0, 1, nv, dtype=np.float32)) * (nv - 1)
    MX, MY = (a.astype(np.float32) for a in np.meshgrid(mx, my))
    return np.stack(
        [
            cv2.remap(
                g[c].astype(np.float32),
                MX,
                MY,
                cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )
            for c in range(g.shape[0])
        ]
    )
