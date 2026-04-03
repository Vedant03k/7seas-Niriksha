from PIL import Image, ImageDraw
import piexif

# Create Fake Image
img_fake = Image.new("RGB", (512, 512), (220, 100, 100))
d = ImageDraw.Draw(img_fake)
d.text((100, 250), "AI Generated Face", fill=(255, 255, 255))
exif_dict = {"0th": {piexif.ImageIFD.Software: b"Stable Diffusion v1.5"}}
img_fake.save("FAKE_AI_Face.jpg", exif=piexif.dump(exif_dict))
print("Created FAKE_AI_Face.jpg")

# Create Real Image
img_real = Image.new("RGB", (512, 512), (100, 220, 100))
d2 = ImageDraw.Draw(img_real)
d2.text((150, 250), "Real Human Face", fill=(0, 0, 0))
img_real.save("REAL_Face.jpg")
print("Created REAL_Face.jpg")

