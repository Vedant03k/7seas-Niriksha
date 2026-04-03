from PIL import Image
try:
    from piexif import dump
except ImportError:
    pass # we can do simple if needed, better to download piexif
