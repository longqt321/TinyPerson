from PIL import Image


def letterbox_geometry(width: int, height: int, size: int):
    scale = min(size / width, size / height)
    return scale, (size - round(width * scale)) // 2, (size - round(height * scale)) // 2


def letterbox(image: Image.Image, size: int):
    width, height = image.size
    scale, left, top = letterbox_geometry(width, height, size)
    new_width, new_height = round(width * scale), round(height * scale)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(image.resize((new_width, new_height), Image.Resampling.BILINEAR), (left, top))
    return canvas, scale, left, top


def map_box(box, scale, left, top):
    x1, y1, x2, y2 = box
    return [x1 * scale + left, y1 * scale + top, x2 * scale + left, y2 * scale + top]
