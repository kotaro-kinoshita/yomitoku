import cv2
import numpy as np
import pytest
import torch

from yomitoku.base import BaseModule
from yomitoku.page_dewarper import PageDewarper
from yomitoku.preprocessor import Preprocessor
from yomitoku.utils.dewarp import (
    content_angle,
    grid_mask,
    grid_stats,
    plane_frontal,
    resample_grid,
    rigid_grid,
    rotate_crop,
    seen_box,
)


def _grid(x0, y0, x1, y1, nv=45, nu=31):
    """Axis-aligned page grid (2, nv, nu) over [x0, x1] x [y0, y1] in normalized photo coordinates."""
    v, u = np.meshgrid(np.linspace(y0, y1, nv), np.linspace(x0, x1, nu), indexing="ij")
    return np.stack([u, v]).astype(np.float32)


def _page(score, g, seen=None):
    if seen is None:
        seen = ((np.abs(g) <= 1).all(0)).astype(np.float32)
    return {"score": score, "grid": g, "seen": seen, "corners": None}


@pytest.fixture(scope="module")
def dewarper():
    return PageDewarper(device="cpu", from_pretrained=False)


def test_grid_stats():
    s = grid_stats(_grid(-0.5, -0.5, 0.5, 0.5))
    assert (
        s["inside"] == 1.0
        and s["flip"] == 0.0
        and s["cell_min"] == pytest.approx(1, rel=1e-3)
    )

    s = grid_stats(_grid(0.5, -0.5, 1.5, 0.5))  # right half outside the photo
    assert 0.4 < s["inside"] < 0.6

    g = _grid(-0.5, -0.5, 0.5, 0.5)
    g[0, :, 20:] = g[0, :, 20:][:, ::-1]  # fold the right part over
    assert grid_stats(g)["flip"] > 0


def test_grid_mask():
    m = grid_mask(_grid(-1, -1, 0, 1), size=64)
    assert m[:, :28].all() and not m[:, 36:].any()


def test_seen_box():
    seen = np.zeros((45, 31), np.float32)
    seen[:, :10] = 1  # only the left third seen
    lo, hi = seen_box(seen)
    assert lo[0] == 0 and 0.25 < hi[0] < 0.4
    assert lo[1] == 0 and hi[1] == 1

    seen[:, :] = 0
    seen[:3, :] = 1  # a few rows seen: with frac=0 every line with a seen node is kept
    lo, hi = seen_box(seen)
    assert hi[1] < 0.1
    seen[:, :] = 1
    seen[:, -1] = 0
    seen[0, -1] = 1  # last column barely seen: kept with frac=0, dropped with frac=0.5
    assert seen_box(seen)[1][0] == 1
    assert seen_box(seen, frac=0.5)[1][0] < 1


def test_resample_grid():
    g = _grid(-1, -1, 1, 1)
    r = resample_grid(g, np.array([0.5, 0.0]), np.array([1.0, 1.0]))
    assert r.shape == g.shape
    assert r[0].min() == pytest.approx(0, abs=1e-4)
    assert r[0].max() == pytest.approx(1, abs=1e-4)


def test_filter(dewarper):
    main = _page(0.99, _grid(-0.9, -0.9, 0.0, 0.9))
    spread = _page(0.98, _grid(0.0, -0.9, 0.9, 0.9))  # the other page of a spread
    nested = _page(0.6, _grid(-0.5, -0.5, -0.2, 0.0))  # inside the main page
    cut = _page(0.7, _grid(0.9, -0.9, 2.5, 0.9))  # mostly outside the photo
    folded = _page(0.8, _grid(-0.9, -0.9, 0.0, 0.9))
    folded["grid"][0, :, 15:] = folded["grid"][0, :, 15:][:, ::-1]

    kept = dewarper.filter([main, spread, nested, cut, folded])
    assert [p["score"] for p in kept] == [0.99, 0.98]

    # the best page is kept even when cut off by the frame
    kept = dewarper.filter([_page(0.9, _grid(0.9, -0.9, 2.5, 0.9))])
    assert len(kept) == 1

    # a larger page wins over pages inside it, whatever the scores
    big = _page(0.4, _grid(-0.9, -0.9, 0.9, 0.9))
    halves = [
        _page(0.6, _grid(-0.9, -0.9, 0.0, 0.9)),
        _page(0.5, _grid(0.0, -0.9, 0.9, 0.9)),
    ]
    assert [p["score"] for p in dewarper.filter(halves + [big])] == [0.4]


def test_pages_order(dewarper, monkeypatch):
    right = _page(0.99, _grid(0.05, -0.9, 0.9, 0.9))
    left = _page(0.98, _grid(-0.9, -0.9, -0.05, 0.9))
    monkeypatch.setattr(dewarper, "detect", lambda img: [right, left])
    assert [p["score"] for p in dewarper.pages(None)] == [0.98, 0.99]


def test_call(dewarper, monkeypatch):
    img = np.full((400, 300, 3), 200, np.uint8)
    left = _page(0.98, _grid(-0.9, -0.9, -0.05, 0.9))
    right = _page(0.99, _grid(0.05, -0.9, 1.4, 0.9))  # partly outside the photo
    monkeypatch.setattr(dewarper, "detect", lambda img: [right, left])
    outputs = dewarper(img)
    assert len(outputs) == 2
    schemas = [s for s, _ in outputs]
    assert [s.page for s in schemas] == [0, 1]
    assert all(s.num_pages == 2 and s.status == "ok" for s in schemas)
    assert len(dewarper.last_results) == 2
    assert dewarper.last_dewarped_img is outputs[0][1]
    for s, flat in outputs:
        assert s.size == [flat.shape[1], flat.shape[0]]
    assert dewarper.visualize(img).shape == img.shape

    monkeypatch.setattr(dewarper, "detect", lambda img: [])
    ((s, out),) = dewarper(img)
    assert s.status == "rejected" and s.reasons == ["no_page"] and out is img


def test_model_forward(dewarper):
    with torch.no_grad():
        heat, off, feats = dewarper.model(torch.zeros(1, 3, 712, 488))
        assert heat.shape == (1, 1, 89, 61) and off.shape == (1, 8, 89, 61)
        corners = torch.tensor(
            [[[-0.8, -0.9], [0.7, -0.85], [0.75, 0.9], [-0.85, 0.8]]]
        )
        g2, g3, seen = dewarper.model.roi(
            feats,
            torch.zeros(1, 3, 712, 488),
            torch.zeros(1, dtype=torch.long),
            corners,
        )
    assert (
        g2.shape == (1, 2, 45, 31)
        and g3.shape == (1, 3, 45, 31)
        and seen.shape == (1, 1, 45, 31)
    )
    assert len(dewarper.detect(np.zeros((300, 200, 3), np.uint8))) <= 16


def test_preprocessor_default_is_multi_page(monkeypatch):
    load = BaseModule.load_model
    monkeypatch.setattr(
        BaseModule,
        "load_model",
        lambda self, name, path_cfg, from_pretrained=True: load(
            self, name, path_cfg, from_pretrained=False
        ),
    )
    pre = Preprocessor(device="cpu")
    assert pre.multi_page and pre.dewarper._cfg.hf_hub_repo.endswith("pagenet-v2")
    img = np.full((300, 200, 3), 128, np.uint8)
    pages = pre.pages(img)
    assert len(pages) >= 1 and all(len(p) == 3 for p in pages)
    results, page_img = pre(img)
    assert results["dewarp"].status in ("ok", "rejected")


def _ruled_page(deg, H=900, W=700):
    """Photo of a ruled page (grey desk, white sheet with dark rules) turned by `deg` (cv2, + = counter-clockwise),
    and the page's grid (2, 45, 31) in [-1, 1]."""
    page = np.full((600, 420, 3), 250, np.uint8)
    for y in range(40, 580, 30):
        cv2.line(page, (30, y), (390, y), (30, 30, 30), 2)
    img = np.full((H, W, 3), 90, np.uint8)
    M = cv2.getRotationMatrix2D((210, 300), deg, 1.0)
    M[:, 2] += [W / 2 - 210, H / 2 - 300]
    cv2.warpAffine(
        page, M, (W, H), img, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT
    )
    v, u = np.meshgrid(np.linspace(0, 600, 45), np.linspace(0, 420, 31), indexing="ij")
    P = np.stack([u, v], -1) @ M[:, :2].T + M[:, 2]
    return img, (P / [W, H] * 2 - 1).transpose(2, 0, 1).astype(np.float32)


def test_plane_frontal():
    v, u = np.meshgrid(np.linspace(0, 1, 45), np.linspace(0, 0.7, 31), indexing="ij")
    flat = np.stack([u, v, np.zeros_like(u)])
    fl, tilt = plane_frontal(flat)
    assert fl < 1e-6 and tilt < 1e-3
    curved = np.stack([u, v, 0.1 * np.sin(np.pi * u / 0.7)])
    assert plane_frontal(curved)[0] > 0.01
    tilted = np.stack([u, v * np.cos(0.3), v * np.sin(0.3)])
    assert plane_frontal(tilted)[1] == pytest.approx(np.degrees(0.3), abs=0.1)


def test_rigid_grid_and_rotate_crop():
    img, g = _ruled_page(3.0)
    noisy = g + np.random.default_rng(0).normal(0, 0.002, g.shape).astype(np.float32)
    r = rigid_grid(noisy, img.shape[:2])
    assert np.abs(r - g).max() < 0.004
    out, res, M = rotate_crop(img, r)
    assert np.degrees(np.arctan2(M[0, 1], M[0, 0])) == pytest.approx(-3.0, abs=0.1)
    assert abs(out.shape[1] - 420) <= 3 and abs(out.shape[0] - 600) <= 3
    a = content_angle(out, _grid(-1, -1, 1, 1))  # rows level: no residual angle
    assert a is None or abs(a) <= 0.1
    # the output's corners map back onto the page's corners in the photo
    H, W = img.shape[:2]
    h, w = out.shape[:2]
    corners = (np.stack([g[:, 0, 0], g[:, 0, -1], g[:, -1, -1], g[:, -1, 0]]) + 1) / 2
    back = res.to_photo([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    assert np.abs(back - corners * [W, H]).max() < 3


def test_flat_page_call(dewarper, monkeypatch):
    img, g = _ruled_page(-1.2)
    # the rectangle fit is turned by 0.4 deg too little: the content deskew finds the rest
    g0 = rigid_grid(g, img.shape[:2])
    c = g0.reshape(2, -1).mean(1)[:, None, None]
    H, W = img.shape[:2]
    P = (g0 - c) * np.array([W, H])[:, None, None]
    t = np.radians(0.4)
    P = np.stack(
        [np.cos(t) * P[0] + np.sin(t) * P[1], -np.sin(t) * P[0] + np.cos(t) * P[1]]
    )
    gt = (P / np.array([W, H])[:, None, None] + c).astype(np.float32)
    v, u = np.meshgrid(np.linspace(0, 1, 45), np.linspace(0, 0.7, 31), indexing="ij")
    page = _page(0.99, gt)
    page["grid3d"] = np.stack([u, v, np.zeros_like(u)]).astype(np.float32)
    monkeypatch.setattr(dewarper, "detect", lambda img: [page])
    ((s, out),) = dewarper(img)
    p = dewarper.last_pages[0]
    assert s.flat is True and p["flat"] and p["deskew_src"] == "content"
    assert s.rotation == pytest.approx(1.2, abs=0.1)
    assert "flat" in s.model_dump() and "rotation" in s.model_dump()
    assert dewarper.visualize(img).shape == img.shape

    # a curved page (3D grid out of plane) is unwarped through its grid
    page["grid3d"][2] = 0.1 * np.sin(np.pi * u / 0.7)
    ((s, out),) = dewarper(img)
    assert s.flat is None and "flat" not in s.model_dump()


def test_sdf_head(dewarper):
    with torch.no_grad():
        feats = dewarper.model.features(torch.zeros(1, 3, 712, 488))
        assert dewarper.model.sdf_map(feats).shape == (1, 1, 178, 122)
