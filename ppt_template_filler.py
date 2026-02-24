# ============================================================
# file: ppt_template_filler.py
# ============================================================
from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import matplotlib.figure
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE_TYPE


_TOKEN_RE = re.compile(r"\{[^}]+\}")


def _hex_to_rgb(hex_color: str) -> RGBColor:
    s = (hex_color or "").strip().lstrip("#")
    if len(s) != 6:
        return RGBColor(255, 255, 255)
    return RGBColor(int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def fig_to_png_bytes(fig: matplotlib.figure.Figure, *, dpi: int = 220) -> bytes:
    bio = io.BytesIO()
    fig.savefig(bio, format="png", dpi=dpi, bbox_inches="tight", transparent=True)
    return bio.getvalue()


def _iter_shapes(slide) -> Iterable:
    for shp in slide.shapes:
        yield shp


def _shape_text(shp) -> str:
    if not getattr(shp, "has_text_frame", False):
        return ""
    try:
        return shp.text_frame.text or ""
    except Exception:
        return ""


def _replace_text_in_shape(shp, replacements: Dict[str, str]) -> None:
    if not getattr(shp, "has_text_frame", False):
        return
    tf = shp.text_frame
    for para in tf.paragraphs:
        for run in para.runs:
            txt = run.text
            if not txt:
                continue
            for k, v in replacements.items():
                if k in txt:
                    txt = txt.replace(k, v)
            run.text = txt
    tf.text = tf.text  # nudge refresh


def _delete_shape(shp) -> None:
    try:
        el = shp.element
        el.getparent().remove(el)
    except Exception:
        pass


def _find_shapes_with_token(slide, token: str) -> List:
    hits = []
    for shp in _iter_shapes(slide):
        if token in _shape_text(shp):
            hits.append(shp)
    return hits


def _set_slide_background(slide, *, bg_hex: str) -> None:
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = _hex_to_rgb(bg_hex)


def _set_shape_fill(shp, *, hex_color: str) -> None:
    try:
        fill = shp.fill
        fill.solid()
        fill.fore_color.rgb = _hex_to_rgb(hex_color)
    except Exception:
        pass


def _is_bar_candidate(shp, slide_w: int, slide_h: int) -> bool:
    """
    Heuristic: wide rectangle spanning most of slide width, near top or bottom.
    """
    try:
        if shp.shape_type not in (MSO_SHAPE_TYPE.AUTO_SHAPE, MSO_SHAPE_TYPE.FREEFORM):
            return False

        w = int(shp.width)
        h = int(shp.height)
        x = int(shp.left)
        y = int(shp.top)

        if w < int(slide_w * 0.85):
            return False
        if h > int(slide_h * 0.12):
            return False

        near_top = y <= int(slide_h * 0.08)
        near_bottom = (y + h) >= int(slide_h * 0.92)

        return (near_top or near_bottom) and x <= int(slide_w * 0.08)
    except Exception:
        return False


def _color_top_and_bottom_bars(slide, *, bar_hex: str, slide_w: int, slide_h: int) -> None:
    # Color any top/bottom “bar-like” shapes
    for shp in _iter_shapes(slide):
        if _is_bar_candidate(shp, slide_w, slide_h):
            _set_shape_fill(shp, hex_color=bar_hex)

    # Also color the specific bottom-bar placeholder shape if present (keep its text)
    for shp in _find_shapes_with_token(slide, "{bottom_bar}"):
        _set_shape_fill(shp, hex_color=bar_hex)


def _insert_image_over_shape(slide, shp, img_bytes: bytes) -> None:
    left, top, width, height = shp.left, shp.top, shp.width, shp.height
    _delete_shape(shp)
    slide.shapes.add_picture(io.BytesIO(img_bytes), left, top, width=width, height=height)


def _fill_token_with_image(slide, token: str, images: List[bytes]) -> None:
    shapes = _find_shapes_with_token(slide, token)
    if not shapes or not images:
        return

    # stable ordering to match left->right placement if multiple placeholders share token
    shapes_sorted = sorted(shapes, key=lambda s: (int(s.left), int(s.top)))
    for shp, img in zip(shapes_sorted, images):
        _insert_image_over_shape(slide, shp, img)


def _find_tables(slide) -> List:
    return [shp.table for shp in _iter_shapes(slide) if getattr(shp, "has_table", False)]


def _write_df_to_ppt_table(table, df: pd.DataFrame) -> None:
    if df is None:
        df = pd.DataFrame()

    n_rows = len(table.rows)
    n_cols = len(table.columns)

    # Clear body
    for r in range(1, n_rows):
        for c in range(n_cols):
            table.cell(r, c).text = ""

    if df.empty:
        return

    values = df.astype(str).replace({"nan": "-", "None": "-"}).values.tolist()
    max_write = max(0, n_rows - 1)

    for r in range(min(max_write, len(values))):
        row_vals = values[r]
        for c in range(min(n_cols, len(row_vals))):
            table.cell(r + 1, c).text = row_vals[c]


@dataclass(frozen=True)
class FilledPptPayload:
    pptx_bytes: bytes
    filename: str


def fill_corner_template_pptx(
    *,
    template_pptx_path: str,
    team_name: str,
    team_primary_hex: str,
    team_secondary_hex: str,
    logo_path: Optional[str],
    meta_replacements: Dict[str, str],
    images_by_token: Dict[str, List[bytes]],
    left_takers_df: pd.DataFrame,
    right_takers_df: pd.DataFrame,
) -> FilledPptPayload:
    """
    - Slide background: secondary color
    - Top+bottom bars: primary color
    - {LOGO}: replaced by image
    - plot tokens: replaced by images
    - Slide1 first table: left takers; Slide2 first table: right takers
    - {bottom_bar} text is not replaced; its shape is only recolored
    """
    if not os.path.exists(template_pptx_path):
        raise FileNotFoundError(f"Template not found: {template_pptx_path}")

    prs = Presentation(template_pptx_path)
    slide_w = int(prs.slide_width)
    slide_h = int(prs.slide_height)

    base_repl = dict(meta_replacements or {})
    base_repl.setdefault("{TEAM_NAME}", team_name)
    base_repl.setdefault("{{TEAM_NAME}}", team_name)

    for slide in prs.slides:
        _set_slide_background(slide, bg_hex=team_secondary_hex)
        _color_top_and_bottom_bars(slide, bar_hex=team_primary_hex, slide_w=slide_w, slide_h=slide_h)

        for shp in _iter_shapes(slide):
            _replace_text_in_shape(shp, base_repl)

        # Logo
        if logo_path and os.path.exists(logo_path):
            with open(logo_path, "rb") as f:
                logo_bytes = f.read()
            for shp in _find_shapes_with_token(slide, "{LOGO}"):
                _insert_image_over_shape(slide, shp, logo_bytes)

        # Images
        for token, imgs in images_by_token.items():
            if token == "{bottom_bar}":
                continue
            _fill_token_with_image(slide, token, imgs)

    # Tables
    if len(prs.slides) >= 1:
        t1 = _find_tables(prs.slides[0])
        if t1:
            _write_df_to_ppt_table(t1[0], left_takers_df)

    if len(prs.slides) >= 2:
        t2 = _find_tables(prs.slides[1])
        if t2:
            _write_df_to_ppt_table(t2[0], right_takers_df)

    out = io.BytesIO()
    prs.save(out)
    fname = f"OpponentAnalysis_Corners_{team_name.replace(' ', '_')}.pptx"
    return FilledPptPayload(pptx_bytes=out.getvalue(), filename=fname)
