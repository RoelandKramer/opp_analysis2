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


def fig_to_png_bytes(fig: matplotlib.figure.Figure, *, dpi: int = 240) -> bytes:
    """
    Full-canvas export so PPT pictures fill placeholder rectangles.
    """
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
        bbox_inches=None,  # critical: don't trim
        pad_inches=0,
    )
    return bio.getvalue()


# ---------------- GROUP-safe traversal w/ absolute coords ----------------
ShapeAbs = Tuple[object, int, int]


def iter_shapes_abs(container, *, off_x: int = 0, off_y: int = 0) -> Iterable[ShapeAbs]:
    for shp in container.shapes:
        left = int(shp.left) + off_x
        top = int(shp.top) + off_y
        yield shp, left, top
        if shp.shape_type == MSO_SHAPE_TYPE.GROUP:
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

    for para in tf.paragraphs:
        for run in para.runs:
            txt = run.text or ""
            for k, v in replacements.items():
                if k in txt:
                    txt = txt.replace(k, v)
            run.text = txt

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
        w, h = int(shp.width), int(shp.height)
        return (
            abs_left <= int(slide_w * 0.05)
            and abs_top <= int(slide_h * 0.05)
            and w >= int(slide_w * 0.90)
            and h >= int(slide_h * 0.90)
        )
    except Exception:
        return False


def _color_background_rectangles(slide, *, bg_hex: str, slide_w: int, slide_h: int) -> None:
    for shp, ax, ay in iter_shapes_abs(slide):
        if _is_full_slide_bg(ax, ay, shp, slide_w, slide_h):
            _set_shape_fill(shp, hex_color=bg_hex)


def _is_bar_candidate(abs_left: int, abs_top: int, shp, slide_w: int, slide_h: int) -> bool:
    try:
        w, h = int(shp.width), int(shp.height)
        if w < int(slide_w * 0.80):
            return False
        if h > int(slide_h * 0.20):
            return False

        near_top = abs_top <= int(slide_h * 0.14)
        near_bottom = (abs_top + h) >= int(slide_h * 0.86)
        return (near_top or near_bottom) and abs_left <= int(slide_w * 0.14)
    except Exception:
        return False


def _find_shapes_with_token(slide, token: str) -> List[ShapeAbs]:
    out: List[ShapeAbs] = []
    for shp, ax, ay in iter_shapes_abs(slide):
        if token in _shape_text(shp):
            out.append((shp, ax, ay))
    return out


def _color_top_and_bottom_bars(slide, *, bar_hex: str, slide_w: int, slide_h: int) -> None:
    # heuristic pass
    for shp, ax, ay in iter_shapes_abs(slide):
        if _is_bar_candidate(ax, ay, shp, slide_w, slide_h):
            _set_shape_fill(shp, hex_color=bar_hex)

    # FORCE: widest shape in top region
    top_region_max_y = int(slide_h * 0.18)
    best_top = None  # (width, -top, shape)
    for shp, ax, ay in iter_shapes_abs(slide):
        try:
            w, h = int(shp.width), int(shp.height)
        except Exception:
            continue
        if ay > top_region_max_y:
            continue
        if w < int(slide_w * 0.65):
            continue
        if h > int(slide_h * 0.30):
            continue
        key = (w, -ay)
        if best_top is None or key > (best_top[0], best_top[1]):
            best_top = (w, -ay, shp)
    if best_top is not None:
        _set_shape_fill(best_top[2], hex_color=bar_hex)

    # ensure bottom placeholder colored (keep text)
    for shp, _, _ in _find_shapes_with_token(slide, "{bottom_bar}"):
        _set_shape_fill(shp, hex_color=bar_hex)


def _set_header_text_white(slide, *, slide_h: int) -> None:
    threshold = int(slide_h * 0.16)
    for shp, _, abs_top in iter_shapes_abs(slide):
        if getattr(shp, "has_text_frame", False) and abs_top <= threshold:
            _set_text_white(shp)


# --------- CRITICAL: use GROUP sibling (Freeform) as target box ----------
Target = Tuple[object, int, int, int, int]  # (token_shape_to_delete, left, top, w, h)


def _target_box_for_token_shape(shp, abs_left: int, abs_top: int) -> Tuple[int, int, int, int]:
    """
    If token is inside a group, use largest non-text sibling (usually Freeform) as the target rect.
    Otherwise use the token shape itself.
    """
    try:
        parent = shp._parent  # GroupShapes or SlideShapes
        group = getattr(parent, "_parent", None)
        if group is not None and getattr(group, "shape_type", None) == MSO_SHAPE_TYPE.GROUP:
            # find largest non-text sibling in this group
            best = None  # (area, sib)
            for sib in group.shapes:
                if getattr(sib, "has_text_frame", False):
                    continue
                try:
                    area = int(sib.width) * int(sib.height)
                except Exception:
                    continue
                if best is None or area > best[0]:
                    best = (area, sib)

            if best is not None:
                sib = best[1]
                # sibling coords are relative to group; compute absolute
                group_left = abs_left - int(shp.left)
                group_top = abs_top - int(shp.top)
                target_left = group_left + int(sib.left)
                target_top = group_top + int(sib.top)
                return target_left, target_top, int(sib.width), int(sib.height)
    except Exception:
        pass

    return abs_left, abs_top, int(shp.width), int(shp.height)


def _find_token_targets(slide, token: str) -> List[Target]:
    out: List[Target] = []
    for shp, ax, ay in _find_shapes_with_token(slide, token):
        left, top, w, h = _target_box_for_token_shape(shp, ax, ay)
        out.append((shp, left, top, w, h))
    # left->right, top->bottom
    return sorted(out, key=lambda t: (t[1], t[2]))


def _insert_picture_fill(slide, *, delete_shape: object, left: int, top: int, w: int, h: int, img_bytes: bytes) -> None:
    """
    Insert picture EXACTLY filling the target rectangle.
    (We stretch to fill; no cropping needed.)
    """
    _delete_shape(delete_shape)
    slide.shapes.add_picture(io.BytesIO(img_bytes), left, top, width=w, height=h)


def _fill_token_with_images(slide, token: str, images: List[bytes]) -> None:
    targets = _find_token_targets(slide, token)
    if not targets or not images:
        return
    for (del_shp, left, top, w, h), img in zip(targets, images):
        _insert_picture_fill(slide, delete_shape=del_shp, left=left, top=top, w=w, h=h, img_bytes=img)


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

    for slide in prs.slides:
        _set_slide_background(slide, bg_hex=team_secondary_hex)
        _color_background_rectangles(slide, bg_hex=team_secondary_hex, slide_w=slide_w, slide_h=slide_h)
        _color_top_and_bottom_bars(slide, bar_hex=team_primary_hex, slide_w=slide_w, slide_h=slide_h)

        for shp, _, _ in iter_shapes_abs(slide):
            _replace_text_in_shape(shp, base_repl)

        _set_header_text_white(slide, slide_h=slide_h)

        # LOGO fills its real block (group sibling freeform)
        if logo_path and os.path.exists(logo_path):
            with open(logo_path, "rb") as f:
                logo_bytes = f.read()
            _fill_token_with_images(slide, "{LOGO}", [logo_bytes])

        # Figures fill their real blocks (group sibling freeform)
        for token, imgs in images_by_token.items():
            if token == "{bottom_bar}":
                continue
            _fill_token_with_images(slide, token, imgs)

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
