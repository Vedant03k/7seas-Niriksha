import os
from reportlab.lib.utils import ImageReader

def add_watermark(c, width, height):
    c.saveState()
    
    # Force a visibly darker gray
    c.setFillColorRGB(0.6, 0.6, 0.6)
    try:
        # Increased opacity to make it darker/more prominent
        c.setFillAlpha(0.22)
    except AttributeError:
        pass

    # Orient to the center of the page and slope upward 45 degrees
    c.translate(width / 2.0, height / 2.0)
    c.rotate(45)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logo_path = os.path.join(base_dir, "frontend", "src", "assets", "nirikshalogo.png")
    
    # We load image logic beforehand
    has_logo = os.path.exists(logo_path)
    img = None
    if has_logo:
        try:
            img = ImageReader(logo_path)
        except Exception:
            has_logo = False

    font_size = 40
    c.setFont("Helvetica-Bold", font_size)
    text = "Niriksha AI"

    # Evaluate lengths to calculate proper repeating spacing
    text_width = c.stringWidth(text, "Helvetica-Bold", font_size)
    logo_size = 45
    spacing = 15

    group_width = text_width + (logo_size + spacing if has_logo else 0)
    
    # Distance between instances (reduced for a tighter grid)
    x_space = group_width + 80
    y_space = font_size + 90

    # Ensure our drawing boundaries go miles beyond the edges of the page,
    # so rotating doesn't leave the corners blank.
    max_dist = max(width, height) * 2.0
    start_x = -max_dist
    end_x = max_dist
    start_y = -max_dist
    end_y = max_dist

    # Draw the dynamic grid
    y = start_y
    row_offset = 0

    while y < end_y:
        # Stagger every other row
        x = start_x + (row_offset % 2) * (x_space / 2.0)
        while x < end_x:
            current_x = x
            if has_logo and img:
                c.drawImage(img, current_x, y - (logo_size * 0.25), width=logo_size, height=logo_size, mask='auto')
                current_x += logo_size + spacing
            c.drawString(current_x, y, text)
            x += x_space
            
        y += y_space
        row_offset += 1

    c.restoreState()