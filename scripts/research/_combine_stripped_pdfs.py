"""
Landscape 2x3 paw figure:
  Row 1: Hind Left  [SNTX print | Naive print | Mean contour]
  Row 2: Hind Right [SNTX print | Naive print | Mean contour]

The vertical centerline of each column lines up; mean-contour cells
share width with paw-print cells.
"""
from __future__ import annotations
import json, time, urllib.request, uuid
from pathlib import Path

import fitz

EXPORT  = Path(r"C:\Users\Gereau\Desktop\Toexport")
SRC_DIR = EXPORT / "stripped"
WEBHOOK = ("")

MEAN_HL_PNG_NAME = "sntx_mean_contour_stripped_HL.png"
MEAN_HR_PNG_NAME = "sntx_mean_contour_stripped_HR.png"

GROUP_ORDER = ["SNTX", "Naive"]
ROW_ORDER   = [("HL", "Hind Left"), ("HR", "Hind Right")]

# Layout (PDF points). 2x3 grid: 3 columns wide (SNTX, Naive, Mean).
PAW_W       = 188
PAW_H       = 240
MEAN_W      = PAW_W
MEAN_H      = PAW_H
# Shrinks the paw-print silhouette within its cell so it visually
# matches the size of the inner outline in the mean-contour panel
# (which is ~65-70% of the cell because of the SD envelope padding).
PAW_INNER_SCALE = 0.65
COL_GAP     = 4
ROW_GAP     = 4
TOP_TITLE_H = 28
LEFT_TITLE_W = 84
PAD_L = 6; PAD_R = 6; PAD_T = 4; PAD_B = 6
GRID_COLOR  = (0.6, 0.6, 0.6)
GRID_WIDTH  = 0.6

TITLE_FONT_SZ = 18
ROW_FONT_SZ   = 16


def step(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def discord_upload(content, files):
    boundary = "----PP" + uuid.uuid4().hex
    body = [f"--{boundary}".encode(),
            b'Content-Disposition: form-data; name="payload_json"',
            b'Content-Type: application/json', b"",
            json.dumps({"content": content}).encode()]
    for i, p in enumerate(files):
        mime = {".png": "image/png", ".pdf": "application/pdf"
                }.get(p.suffix.lower(), "application/octet-stream")
        body += [f"--{boundary}".encode(),
                 f'Content-Disposition: form-data; name="file{i}"; filename="{p.name}"'.encode(),
                 f"Content-Type: {mime}".encode(), b"",
                 p.read_bytes()]
    body.append(f"--{boundary}--".encode())
    req = urllib.request.Request(
        WEBHOOK, data=b"\r\n".join(body), method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                 "User-Agent": "PixelPaws-Combine2x3Landscape/1.0"})
    with urllib.request.urlopen(req, timeout=180) as r:
        step(f"  HTTP {r.status}")


def _is_white(fill) -> bool:
    if fill is None: return False
    return all(abs(c - 1.0) < 0.02 for c in fill[:3])


def filled_bbox(page: fitz.Page, clip: fitz.Rect) -> fitz.Rect | None:
    bbox = None
    for d in page.get_drawings():
        fill = d.get("fill")
        if fill is None or _is_white(fill):
            continue
        rect = d.get("rect")
        if rect is None:
            continue
        inter = rect & clip
        if inter.is_empty:
            continue
        bbox = rect if bbox is None else bbox | rect
    return bbox


def split_into_two(src_page) -> tuple[fitz.Rect, fitz.Rect]:
    r = src_page.rect
    midx = (r.x0 + r.x1) / 2
    return (fitz.Rect(r.x0, r.y0, midx, r.y1),
            fitz.Rect(midx, r.y0, r.x1, r.y1))


def fit_rect(bb_w: float, bb_h: float, dest: fitz.Rect) -> fitz.Rect:
    dw, dh = dest.width, dest.height
    scale = min(dw / bb_w, dh / bb_h)
    new_w = bb_w * scale
    new_h = bb_h * scale
    cx = (dest.x0 + dest.x1) / 2
    cy = (dest.y0 + dest.y1) / 2
    return fitz.Rect(cx - new_w / 2, cy - new_h / 2,
                     cx + new_w / 2, cy + new_h / 2)


def insert_centred_text(page: fitz.Page, text: str,
                        centre: tuple[float, float], fontsize: int,
                        fontname: str = "helvetica-bold") -> None:
    text_w = fontsize * 0.55 * len(text)
    x = centre[0] - text_w / 2
    y = centre[1] + fontsize / 3
    page.insert_text((x, y), text, fontsize=fontsize, fontname=fontname,
                     color=(0, 0, 0))


def main():
    hl = SRC_DIR / "contour_print_HL_stripped.pdf"
    hr = SRC_DIR / "contour_print_HR_stripped.pdf"
    mean_hl_png = EXPORT / MEAN_HL_PNG_NAME
    mean_hr_png = EXPORT / MEAN_HR_PNG_NAME
    for p in (hl, hr, mean_hl_png, mean_hr_png):
        if not p.exists():
            step(f"! missing {p}"); return 1

    src = {"HL": fitz.open(hl), "HR": fitz.open(hr)}

    panel_bbox = {}
    for role in ("HL", "HR"):
        page = src[role][0]
        left, right = split_into_two(page)
        for group, clip in zip(GROUP_ORDER, (left, right)):
            bb = filled_bbox(page, clip)
            if bb is None:
                step(f"! no filled paths in {role} {group}"); return 1
            panel_bbox[(role, group)] = bb

    # Layout: 3 columns (SNTX, Naive, Mean), 2 rows (HL, HR)
    out_w = (PAD_L + LEFT_TITLE_W + PAW_W + COL_GAP + PAW_W
             + COL_GAP + MEAN_W + PAD_R)
    out_h = PAD_T + TOP_TITLE_H + PAW_H + ROW_GAP + PAW_H + PAD_B

    out = fitz.open()
    new_page = out.new_page(width=out_w, height=out_h)

    # Column x positions
    col_x = {
        "SNTX":  PAD_L + LEFT_TITLE_W,
        "Naive": PAD_L + LEFT_TITLE_W + PAW_W + COL_GAP,
        "Mean":  PAD_L + LEFT_TITLE_W + PAW_W + COL_GAP + PAW_W + COL_GAP,
    }
    # Column titles
    title_y = PAD_T + TOP_TITLE_H / 2
    insert_centred_text(new_page, "SNTX",  (col_x["SNTX"] + PAW_W / 2, title_y),
                         TITLE_FONT_SZ)
    insert_centred_text(new_page, "Naive", (col_x["Naive"] + PAW_W / 2, title_y),
                         TITLE_FONT_SZ)
    insert_centred_text(new_page, "Mean Contour",
                         (col_x["Mean"] + MEAN_W / 2, title_y),
                         TITLE_FONT_SZ)

    # Row y positions + labels
    row_y = {"HL": PAD_T + TOP_TITLE_H, "HR": PAD_T + TOP_TITLE_H + PAW_H + ROW_GAP}
    row_x = PAD_L + LEFT_TITLE_W / 2
    insert_centred_text(new_page, "Hind Left",
                         (row_x, row_y["HL"] + PAW_H / 2), ROW_FONT_SZ)
    insert_centred_text(new_page, "Hind Right",
                         (row_x, row_y["HR"] + PAW_H / 2), ROW_FONT_SZ)

    # Grid lines: light grey borders around each of the six cells
    for role, _ in ROW_ORDER:
        for col_name in ("SNTX", "Naive", "Mean"):
            x0 = col_x[col_name]
            cell_w = MEAN_W if col_name == "Mean" else PAW_W
            cell_h = MEAN_H if col_name == "Mean" else PAW_H
            cell = fitz.Rect(x0, row_y[role], x0 + cell_w, row_y[role] + cell_h)
            new_page.draw_rect(cell, color=GRID_COLOR, width=GRID_WIDTH)

    # Paw cells -- silhouette shrunk to PAW_INNER_SCALE of cell to match
    # the visual size of the mean-contour outline inside its envelope.
    for role, _ in ROW_ORDER:
        for group in GROUP_ORDER:
            cell = fitz.Rect(col_x[group], row_y[role],
                             col_x[group] + PAW_W, row_y[role] + PAW_H)
            # Shrink the destination cell so the paw doesn't fill it
            cx = (cell.x0 + cell.x1) / 2
            cy = (cell.y0 + cell.y1) / 2
            sw = cell.width * PAW_INNER_SCALE
            sh = cell.height * PAW_INNER_SCALE
            inner_cell = fitz.Rect(cx - sw / 2, cy - sh / 2,
                                    cx + sw / 2, cy + sh / 2)
            bb = panel_bbox[(role, group)]
            dest = fit_rect(bb.width, bb.height, inner_cell)
            new_page.show_pdf_page(dest, src[role], 0, clip=bb)

    # Mean-contour cells
    import struct
    def _png_dims(p: Path) -> tuple[int, int]:
        with open(p, "rb") as f:
            head = f.read(24)
        w, h = struct.unpack(">II", head[16:24])
        return w, h

    for role, png in (("HL", mean_hl_png), ("HR", mean_hr_png)):
        cell = fitz.Rect(col_x["Mean"], row_y[role],
                         col_x["Mean"] + MEAN_W, row_y[role] + MEAN_H)
        w, h = _png_dims(png)
        dest = fit_rect(w, h, cell)
        new_page.insert_image(dest, filename=str(png))

    out_pdf = SRC_DIR / "contour_print_6panel_landscape.pdf"
    out.save(out_pdf, deflate=True, garbage=4)

    zoom = 240 / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix_w = new_page.get_pixmap(matrix=mat, alpha=False)
    pix_t = new_page.get_pixmap(matrix=mat, alpha=True)
    out_png_w = SRC_DIR / "contour_print_6panel_landscape_white.png"
    out_png_t = SRC_DIR / "contour_print_6panel_landscape.png"
    pix_w.save(out_png_w)
    pix_t.save(out_png_t)
    out.close()
    for d in src.values(): d.close()
    step(f"wrote {out_pdf.name}, {out_png_w.name}, {out_png_t.name}")

    head = ("**Six-panel paw figure -- landscape 2x3**\n"
            "Rows = Hind Left / Hind Right; columns = SNTX print, Naive print, "
            "Mean Contour. Each row's paw prints and mean contour are inline "
            "so the HL and HR examples sit next to their averaged shape. PDF + "
            "white-bg PNG + transparent PNG.")
    discord_upload(head, [out_pdf, out_png_w, out_png_t])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
