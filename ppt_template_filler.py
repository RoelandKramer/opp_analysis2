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
    Full-canvas export. Avoid bbox_inches='tight' to prevent shrinking.
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
        bbox_inches=None,
        pad_inches=0,
    )
    return bio.getvalue()


# ---------------- GROUP-mapped traversal (handles scaling!) ----------------
ShapeMapped = Tuple[object, int, int, int, int, Optional[object]]
# (shape, abs_left, abs_top, abs_w, abs_h, parent_group_shape)


def _children_bbox(group_shape) -> Tuple[int, int, int, int]:
    """
    Bounding box in the GROUP's internal coordinate system.
    """
    min_l, min_t = 10**18, 10**18
    max_r, max_b = -10**18, -10**18

    for ch in group_shape.shapes:
        l, t = int(ch.left), int(ch.top)
        r, b = l + int(ch.width), t + int(ch.height)
        min_l = min(min_l, l)
        min_t = min(min_t, t)
        max_r = max(max_r, r)
        max_b = max(max_b, b)

    if min_l == 10**18:
        return 0, 0, 1, 1
    return min_l, min_t, max_r, max_b


def iter_shapes_mapped(container, *, base_left: int = 0, base_top: int = 0, base_sx: float = 1.0, base_sy: float = 1.0, parent_group: Optional[object] = None) -> Iterable[ShapeMapped]:
    """
    Yields shapes with *true slide* coordinates, including shapes inside scaled groups.
    """
    # container can be a slide or a group shape (both have .shapes)
    for shp in container.shapes:
        # First map this shape's left/top/size using current transform
        rel_l, rel_t = int(shp.left), int(shp.top)
        rel_w, rel_h = int(shp.width), int(shp.height)

        abs_left = int(round(base_left + rel_l * base_sx))
        abs_top = int(round(base_top + rel_t * base_sy))
        abs_w = int(round(rel_w * base_sx))
        abs_h = int(round(rel_h * base_sy))

        yield shp, abs_left, abs_top, abs_w, abs_h, parent_group

        # If it's a group, compute its internal bbox and scaling to slide coords
        if shp.shape_type == MSO_SHAPE_TYPE.GROUP:
            min_l, min_t, max_r, max_b = _children_bbox(shp)
            internal_w = max(1, max_r - min_l)
            internal_h = max(1, max_b - min_t)

            # Group's displayed size in slide coords is abs_w/abs_h
            sx = abs_w / internal_w
            sy = abs_h / internal_h

            # New base maps group-internal coords to slide coords:
            # slide = group_abs_origin + (internal - min) * scale
            child_base_left = abs_left - int(round(min_l * sx))
            child_base_top = abs_top - int(round(min_t * sy))

            yield from iter_shapes_mapped(
                shp,
                base_left=child_base_left,
                base_top=child_base_top,
                base_sx=sx,
                base_sy=sy,
                parent_group=shp,
            )


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

    # Handle token split across runs
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


def _find_shapes_with_token(slide, token: str) -> List[ShapeMapped]:
    out: List[ShapeMapped] = []
    for shp, ax, ay, aw, ah, parent in iter_shapes_mapped(slide):
        if token in _shape_text(shp):
            out.append((shp, ax, ay, aw, ah, parent))
    return out


def _color_background_rectangles(slide, *, bg_hex: str, slide_w: int, slide_h: int) -> None:
    # recolor any full-slide rectangle that acts as "background"
    for shp, ax, ay, aw, ah, _ in iter_shapes_mapped(slide):
        if ax <= int(slide_w * 0.05) and ay <= int(slide_h * 0.05) and aw >= int(slide_w * 0.90) and ah >= int(slide_h * 0.90):
            _set_shape_fill(shp, hex_color=bg_hex)


def _is_bar_candidate(ax: int, ay: int, aw: int, ah: int, slide_w: int, slide_h: int) -> bool:
    if aw < int(slide_w * 0.80):
        return False
    if ah > int(slide_h * 0.20):
        return False
    near_top = ay <= int(slide_h * 0.14)
    near_bottom = (ay + ah) >= int(slide_h * 0.86)
    return (near_top or near_bottom) and ax <= int(slide_w * 0.14)


def _color_top_and_bottom_bars(slide, *, bar_hex: str, slide_w: int, slide_h: int) -> None:
    # heuristic (now using mapped coords)
    for shp, ax, ay, aw, ah, _ in iter_shapes_mapped(slide):
        if _is_bar_candidate(ax, ay, aw, ah, slide_w, slide_h):
            _set_shape_fill(shp, hex_color=bar_hex)

    # force: widest shape in top region
    top_region_max_y = int(slide_h * 0.18)
    best = None  # (aw, -ay, shp)
    for shp, ax, ay, aw, ah, _ in iter_shapes_mapped(slide):
        if ay > top_region_max_y:
            continue
        if aw < int(slide_w * 0.65):
            continue
        if ah > int(slide_h * 0.30):
            continue
        key = (aw, -ay)
        if best is None or key > (best[0], best[1]):
            best = (aw, -ay, shp)
    if best is not None:
        _set_shape_fill(best[2], hex_color=bar_hex)

    # always color bottom bar placeholder shape
    for shp, *_ in _find_shapes_with_token(slide, "{bottom_bar}"):
        _set_shape_fill(shp, hex_color=bar_hex)


def _set_header_text_white(slide, *, slide_h: int) -> None:
    threshold = int(slide_h * 0.16)
    for shp, _, ay, _, _, _ in iter_shapes_mapped(slide):
        if getattr(shp, "has_text_frame", False) and ay <= threshold:
            _set_text_white(shp)


# --------- token -> target box: choose best sibling inside same group ----------
Target = Tuple[object, int, int, int, int]  # (token_shape_to_delete, left, top, w, h)


def _best_sibling_rect_in_group(
    *,
    token_shp: object,
    token_ax: int,
    token_ay: int,
    token_aw: int,
    token_ah: int,
    parent_group: Optional[object],
) -> Tuple[int, int, int, int]:
    """
    In your template, the real block is the large Freeform sibling inside the same GROUP.
    We select the largest-area sibling (excluding the token textbox itself) in mapped coordinates.
    """
    if parent_group is None:
        return token_ax, token_ay, token_aw, token_ah

    # build mapped rects for this group only
    siblings: List[Tuple[int, int, int, int, object]] = []
    for shp, ax, ay, aw, ah, pg in iter_shapes_mapped(parent_group, base_left=0, base_top=0, base_sx=1.0, base_sy=1.0, parent_group=parent_group):
        # iter_shapes_mapped(parent_group) yields the group itself too; skip it
        if shp is parent_group:
            continue
        siblings.append((ax, ay, aw, ah, shp))

    # token shape in group-local mapped coords is unknown; easiest: pick the sibling with max area
    # that is not the token shape and not a textbox (prefer freeform)
    best = None  # (score, ax, ay, aw, ah, shp)
    for ax, ay, aw, ah, shp in siblings:
        if shp is token_shp:
            continue
        area = aw * ah
        shape_type = getattr(shp, "shape_type", None)
        # preference: non-textbox gets boost
        boost = 1_000_000_000 if shape_type != MSO_SHAPE_TYPE.TEXT_BOX else 0
        score = boost + area
        if best is None or score > best[0]:
            best = (score, ax, ay, aw, ah, shp)

    if best is None:
        return token_ax, token_ay, token_aw, token_ah

    # BUT those coords were group-local; we need slide coords.
    # So: compute slide coords by re-finding that sibling on slide traversal.
    # We do this by matching object identity (same python object).
    # Caller will do that; if we can't, fallback to token.
    return token_ax, token_ay, token_aw, token_ah  # placeholder; resolved in _find_token_targets


def _find_token_targets(slide, token: str) -> List[Target]:
    hits = _find_shapes_with_token(slide, token)
    out: List[Target] = []

    for token_shp, token_ax, token_ay, token_aw, token_ah, parent_group in hits:
        # If in group: choose best sibling on slide (mapped coords)
        if parent_group is not None:
            # pick sibling by scanning shapes within that *same parent group* on the slide, using mapped coords
            best = None  # (score, ax, ay, aw, ah, shp)
            for shp, ax, ay, aw, ah, pg in iter_shapes_mapped(slide):
                if pg is not parent_group:
                    continue
                if shp is token_shp:
                    continue
                area = aw * ah
                st = getattr(shp, "shape_type", None)
                boost = 1_000_000_000 if st != MSO_SHAPE_TYPE.TEXT_BOX else 0
                score = boost + area
                if best is None or score > best[0]:
                    best = (score, ax, ay, aw, ah, shp)

            if best is not None:
                _, ax, ay, aw, ah, _ = best
                out.append((token_shp, ax, ay, aw, ah))
            else:
                out.append((token_shp, token_ax, token_ay, token_aw, token_ah))
        else:
            out.append((token_shp, token_ax, token_ay, token_aw, token_ah))

    return sorted(out, key=lambda t: (t[1], t[2]))


def _insert_picture_fill(slide, *, delete_shape: object, left: int, top: int, w: int, h: int, img_bytes: bytes) -> None:
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

        for shp, *_ in iter_shapes_mapped(slide):
            _replace_text_in_shape(shp, base_repl)

        _set_header_text_white(slide, slide_h=slide_h)

        # Logo
        if logo_path and os.path.exists(logo_path):
            with open(logo_path, "rb") as f:
                logo_bytes = f.read()
            _fill_token_with_images(slide, "{LOGO}", [logo_bytes])

        # Figures
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
