# ============================================================
# file: ppt_template_filler.py
# ============================================================
from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.figure
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE_TYPE


def _hex_to_rgb(hex_color: str) -> RGBColor:
    s = (hex_color or "").strip().lstrip("#")
    if len(s) != 6:
        return RGBColor(255, 255, 255)
    return RGBColor(int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def fig_to_png_bytes(fig: matplotlib.figure.Figure, *, dpi: int = 220) -> bytes:
    """
    Export full-canvas PNG so it fills PPT placeholder exactly.
    DO NOT use bbox_inches='tight' (it shrinks to content).
    """
    # Ensure axes use full figure area
    try:
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        for ax in fig.axes:
            ax.set_position([0, 0, 1, 1])
    except Exception:
        pass

    bio = io.BytesIO()
    fig.savefig(
        bio,
        format="png",
        dpi=dpi,
        transparent=True,
        pad_inches=0,
        bbox_inches=None,   # critical
    )
    return bio.getvalue()

# ---------------- GROUP-safe traversal w/ absolute coords ----------------
ShapeWithAbs = Tuple[object, int, int]


def iter_shapes_abs(container, *, off_x: int = 0, off_y: int = 0) -> Iterable[ShapeWithAbs]:
    """
    Yields (shape, abs_left, abs_top) for:
      - slide
      - group shapes (recursively)
    Coordinates are absolute in slide space.
    """
    for shp in container.shapes:
        try:
            left = int(shp.left) + off_x
            top = int(shp.top) + off_y
        except Exception:
            left = off_x
            top = off_y

        yield shp, left, top

        if shp.shape_type == MSO_SHAPE_TYPE.GROUP:
            # Children positions are relative to group.
            yield from iter_shapes_abs(shp, off_x=left, off_y=top)


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
    # Run-level
    for para in tf.paragraphs:
        for run in para.runs:
            txt = run.text
            if not txt:
                continue
            for k, v in replacements.items():
                if k in txt:
                    txt = txt.replace(k, v)
            run.text = txt

    # Paragraph-level (handles token split across runs)
    for para in tf.paragraphs:
        whole = "".join(run.text for run in para.runs)
        replaced = whole
        for k, v in replacements.items():
            replaced = replaced.replace(k, v)
        if replaced != whole:
            for run in list(para.runs):
                run.text = ""
            if para.runs:
                para.runs[0].text = replaced
            else:
                para.add_run().text = replaced


def _set_text_white(shp) -> None:
    if not getattr(shp, "has_text_frame", False):
        return
    for para in shp.text_frame.paragraphs:
        for run in para.runs:
            try:
                run.font.color.rgb = RGBColor(255, 255, 255)
            except Exception:
                pass


def _delete_shape(shp) -> None:
    try:
        el = shp.element
        el.getparent().remove(el)
    except Exception:
        pass


def _find_shapes_with_token(slide, token: str) -> List[ShapeWithAbs]:
    hits: List[ShapeWithAbs] = []
    for shp, ax, ay in iter_shapes_abs(slide):
        if token in _shape_text(shp):
            hits.append((shp, ax, ay))
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


def _is_full_slide_bg(abs_left: int, abs_top: int, shp, slide_w: int, slide_h: int) -> bool:
    try:
        w = int(shp.width)
        h = int(shp.height)
        return (
            abs_left <= int(slide_w * 0.05)
            and abs_top <= int(slide_h * 0.05)
            and w >= int(slide_w * 0.90)
            and h >= int(slide_h * 0.90)
        )
    except Exception:
        return False


def _is_bar_candidate(abs_left: int, abs_top: int, shp, slide_w: int, slide_h: int) -> bool:
    """
    Bar heuristic using ABS coords.
    """
    try:
        if shp.shape_type not in (MSO_SHAPE_TYPE.AUTO_SHAPE, MSO_SHAPE_TYPE.FREEFORM):
            return False

        w = int(shp.width)
        h = int(shp.height)

        if w < int(slide_w * 0.80):
            return False
        if h > int(slide_h * 0.18):
            return False

        near_top = abs_top <= int(slide_h * 0.12)
        near_bottom = (abs_top + h) >= int(slide_h * 0.88)

        return (near_top or near_bottom) and abs_left <= int(slide_w * 0.12)
    except Exception:
        return False


def _color_background_rectangles(slide, *, bg_hex: str, slide_w: int, slide_h: int) -> None:
    for shp, ax, ay in iter_shapes_abs(slide):
        if _is_full_slide_bg(ax, ay, shp, slide_w, slide_h):
            _set_shape_fill(shp, hex_color=bg_hex)


def _color_top_and_bottom_bars(slide, *, bar_hex: str, slide_w: int, slide_h: int) -> None:
    # Color any bar-like shapes (top or bottom) with ABS coords
    for shp, ax, ay in iter_shapes_abs(slide):
        if _is_bar_candidate(ax, ay, shp, slide_w, slide_h):
            _set_shape_fill(shp, hex_color=bar_hex)

    # Also color bottom-bar placeholder if present (keep its text)
    for shp, _, _ in _find_shapes_with_token(slide, "{bottom_bar}"):
        _set_shape_fill(shp, hex_color=bar_hex)


def _insert_image_over_shape(slide, shp, *, abs_left: int, abs_top: int, img_bytes: bytes) -> None:
    """
    For GROUP children: abs_left/abs_top fixes placement.
    """
    width, height = shp.width, shp.height
    _delete_shape(shp)
    slide.shapes.add_picture(
        io.BytesIO(img_bytes),
        abs_left,
        abs_top,
        width=width,
        height=height,
    )


def _fill_token_with_image(slide, token: str, images: List[bytes]) -> None:
    hits = _find_shapes_with_token(slide, token)
    if not hits or not images:
        return

    # Stable ordering: left->right, top->bottom in ABS space
    hits_sorted = sorted(hits, key=lambda t: (t[1], t[2]))

    for (shp, ax, ay), img in zip(hits_sorted, images):
        _insert_image_over_shape(slide, shp, abs_left=ax, abs_top=ay, img_bytes=img)


def _find_tables(slide) -> List:
    return [shp.table for shp in slide.shapes if getattr(shp, "has_table", False)]


def _write_df_to_ppt_table(table, df: pd.DataFrame) -> None:
    if df is None:
        df = pd.DataFrame()

    n_rows = len(table.rows)
    n_cols = len(table.columns)

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


def _set_header_text_white(slide, *, slide_h: int) -> None:
    """
    Make any text shape in top ~16% of slide white (ABS coords).
    """
    threshold = int(slide_h * 0.16)
    for shp, _, abs_top in iter_shapes_abs(slide):
        if getattr(shp, "has_text_frame", False) and abs_top <= threshold:
            _set_text_white(shp)


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
        _color_background_rectangles(slide, bg_hex=team_secondary_hex, slide_w=slide_w, slide_h=slide_h)
        _color_top_and_bottom_bars(slide, bar_hex=team_primary_hex, slide_w=slide_w, slide_h=slide_h)

        # Replace tokens inside GROUPs too
        for shp, _, _ in iter_shapes_abs(slide):
            _replace_text_in_shape(shp, base_repl)

        _set_header_text_white(slide, slide_h=slide_h)

        # Logo (GROUP-safe + ABS placement)
        if logo_path and os.path.exists(logo_path):
            with open(logo_path, "rb") as f:
                logo_bytes = f.read()
            for shp, ax, ay in _find_shapes_with_token(slide, "{LOGO}"):
                _insert_image_over_shape(slide, shp, abs_left=ax, abs_top=ay, img_bytes=logo_bytes)

        # Figures (GROUP-safe + ABS placement)
        for token, imgs in images_by_token.items():
            if token == "{bottom_bar}":
                continue
            _fill_token_with_image(slide, token, imgs)

    # Tables (top-level)
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
