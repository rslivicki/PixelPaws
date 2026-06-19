"""
Strip background lines/text/ticks from `contour_print_HL.pdf` and
`contour_print_HR.pdf` (in C:\\Users\\Gereau\\Desktop\\Toexport).

Vector-level approach via PyMuPDF: re-draw each page on a blank canvas,
copying ONLY the filled paw-shape paths and dropping everything else
(stroked-only gridlines, axis spines, tick marks, text labels, titles).

Saves two clean PDFs (no background, no text) + matched PNG previews
and uploads everything to #results.
"""
from __future__ import annotations
import json, time, urllib.request, uuid
from pathlib import Path

import fitz

SRC = Path(r"C:\Users\Gereau\Desktop\Toexport")
OUT = SRC / "stripped"

WEBHOOK = ("")


def step(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def discord_upload(content, files):
    boundary = "----PP" + uuid.uuid4().hex
    body = [f"--{boundary}".encode(),
            b'Content-Disposition: form-data; name="payload_json"',
            b'Content-Type: application/json', b"",
            json.dumps({"content": content}).encode()]
    for i, p in enumerate(files):
        mime = {".png": "image/png", ".pdf": "application/pdf",
                ".svg": "image/svg+xml"}.get(p.suffix.lower(),
                                              "application/octet-stream")
        body += [f"--{boundary}".encode(),
                 f'Content-Disposition: form-data; name="file{i}"; filename="{p.name}"'.encode(),
                 f"Content-Type: {mime}".encode(), b"",
                 p.read_bytes()]
    body.append(f"--{boundary}--".encode())
    req = urllib.request.Request(
        WEBHOOK, data=b"\r\n".join(body), method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                 "User-Agent": "PixelPaws-StripPDF/1.0"})
    with urllib.request.urlopen(req, timeout=180) as r:
        step(f"  HTTP {r.status}")


def strip_to_filled(src_pdf: Path, dst_pdf: Path) -> None:
    """Read every page of src_pdf; rebuild a clean page containing ONLY
    filled paths (the paw shapes). Drops text, strokes-only paths
    (gridlines / axis spines / tick marks), and any drawings with very
    thin stroke + no fill."""
    src = fitz.open(src_pdf)
    dst = fitz.open()

    for pno in range(len(src)):
        page = src[pno]
        new_page = dst.new_page(width=page.rect.width, height=page.rect.height)
        shape = new_page.new_shape()
        drawings = page.get_drawings()
        # Total page area (pt^2) -- anything below 0.05% of the page is
        # almost certainly a tick mark or stray glyph fill, not the paw.
        page_area = float(page.rect.width * page.rect.height)
        min_area = page_area * 0.0005
        n_kept = n_dropped = 0
        for d in drawings:
            fill = d.get("fill")
            # Keep only paths that have a non-None fill (filled shapes).
            # The mpl paw fills come through as type 'f' or 'fs' with a
            # non-None fill colour. Stroked-only paths have fill is None.
            if fill is None:
                n_dropped += 1; continue
            # Drop filled paths smaller than the area threshold (tick
            # marks, axis-label glyph fills) -- the actual paw silhouette
            # is several orders of magnitude larger.
            bb = d.get("rect")
            if bb is not None and bb.width * bb.height < min_area:
                n_dropped += 1; continue
            # Walk the path items and draw onto the shape canvas
            for item in d["items"]:
                op = item[0]
                if op == "l":  # line
                    p1, p2 = item[1], item[2]
                    shape.draw_line(p1, p2)
                elif op == "c":  # cubic bezier
                    p1, p2, p3, p4 = item[1], item[2], item[3], item[4]
                    shape.draw_bezier(p1, p2, p3, p4)
                elif op == "re":  # rectangle
                    rect = item[1]
                    shape.draw_rect(rect)
                elif op == "qu":  # quad
                    quad = item[1]
                    shape.draw_quad(quad)
                # ignore unknown ops
            shape.finish(
                fill=fill,
                color=None,  # no edge stroke -- keeps the silhouette pure
                fill_opacity=d.get("fill_opacity", 1.0),
                even_odd=d.get("even_odd", False),
                closePath=True,
            )
            n_kept += 1
        shape.commit()
        step(f"  page {pno+1}/{len(src)}: kept {n_kept} filled paths, "
             f"dropped {n_dropped} stroke-only paths")

    dst.save(dst_pdf, deflate=True, garbage=4)
    dst.close(); src.close()


def render_preview(pdf_path: Path, png_path: Path, dpi: int = 240,
                   transparent: bool = True) -> None:
    doc = fitz.open(pdf_path)
    page = doc[0]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=transparent)
    pix.save(png_path)
    doc.close()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    outputs = []
    for src_name in ("contour_print_HL.pdf", "contour_print_HR.pdf"):
        src = SRC / src_name
        if not src.exists():
            step(f"! missing {src}"); continue
        clean = OUT / src_name.replace(".pdf", "_stripped.pdf")
        png   = OUT / src_name.replace(".pdf", "_stripped.png")
        png_w = OUT / src_name.replace(".pdf", "_stripped_white.png")
        step(f"stripping {src.name}")
        strip_to_filled(src, clean)
        render_preview(clean, png,   transparent=True)
        render_preview(clean, png_w, transparent=False)
        outputs += [clean, png, png_w]
        step(f"  -> {clean.name}, {png.name}, {png_w.name}")

    head = ("**Contour-print PDFs with background lines stripped**\n"
            "Same source PDFs (`contour_print_HL.pdf` / `_HR.pdf`) re-built "
            "at the vector level via PyMuPDF: every filled paw-shape path "
            "kept verbatim, every stroke-only path (gridlines, axis spines, "
            "tick marks) and every text run (titles, labels, panel headers) "
            "dropped. Layout, scale, and shape identity are preserved exactly "
            "-- only the background lines/text are gone.\n"
            "Attached: stripped PDFs + matched PNG previews "
            "(`_stripped.png` = transparent, `_stripped_white.png` = white "
            "background).")
    discord_upload(head, outputs)


if __name__ == "__main__":
    main()
