import os
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader

# Resolve logo path once at import time
_LOGO_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "frontend", "src", "assets", "nirikshalogo.png",
)


def add_watermark(c: Canvas, width: float, height: float):
    """
    Add a tiled diagonal 'NIRIKSHA' watermark grid with a centred logo
    to the current PDF page.
    """
    c.saveState()

    # ── Tiled text watermarks ────────────────────────────────────────
    c.setFillColor(colors.Color(0, 0, 0, alpha=0.035))
    c.setFont("Helvetica-Bold", 38)

    spacing_x = 220
    spacing_y = 180

    for x in range(-100, int(width) + 200, spacing_x):
        for y in range(-100, int(height) + 200, spacing_y):
            c.saveState()
            c.translate(x, y)
            c.rotate(40)
            c.drawCentredString(0, 0, "NIRIKSHA")
            c.restoreState()

    # ── Centre logo watermark ────────────────────────────────────────
    if os.path.isfile(_LOGO_PATH):
        try:
            logo = ImageReader(_LOGO_PATH)
            logo_w, logo_h = 120, 120
            lx = (width - logo_w) / 2
            ly = (height - logo_h) / 2
            c.setFillAlpha(0.06)
            c.drawImage(logo, lx, ly, width=logo_w, height=logo_h,
                        preserveAspectRatio=True, mask='auto')
        except Exception:
            pass  # degrade gracefully if logo can't be read

    c.restoreState()
