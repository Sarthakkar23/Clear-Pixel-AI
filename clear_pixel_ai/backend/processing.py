import cv2
import numpy as np
from PIL import Image
import io
import os

def denoise_opencv(image):
    # Use OpenCV's fastNlMeansDenoisingColored as an alternative denoising method
    denoised = cv2.fastNlMeansDenoisingColored(image, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
    return denoised

def restore_image(image):
    # Enhanced restoration for old photos using CLAHE, saturation boost, and sharpening
    img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)

    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    # Merge channels back
    limg = cv2.merge((cl,a,b))
    img_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Convert to HSV to boost saturation
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    s = cv2.add(s, 25)  # Increase saturation by 25 (clipped automatically)
    img_hsv = cv2.merge((h, s, v))
    img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    # Sharpen image
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(img_bgr, -1, kernel)

    return sharpened

def filter_image(image, filter_type):
    # Basic filters using OpenCV
    if filter_type == 'blur':
        return cv2.GaussianBlur(image, (7, 7), 0)
    elif filter_type == 'sharpen':
        kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)
    else:
        return image

def edge_canny(image):
    # Use OpenCV Canny edge detector as alternative
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return edges_bgr

def super_resolve_resize(image, scale=2):
    # Use OpenCV resize with cubic interpolation for super resolution
    height, width = image.shape[:2]
    new_size = (width * scale, height * scale)
    super_resolved = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
    return super_resolved

def inpaint_telea(image, mask=None):
    # Use OpenCV inpainting with TELEA method
    if mask is None:
        return image
    inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    return inpainted

def process_image_bytes(image_bytes, operation=None, filter_type=None, mask_bytes=None):
    # Load image from bytes
    pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = np.array(pil_image)[:, :, ::-1]  # Convert RGB to BGR for OpenCV

    if operation == 'denoise':
        processed = denoise_opencv(image)
    elif operation == 'restore':
        processed = restore_image(image)
    elif operation == 'filter':
        processed = filter_image(image, filter_type)
    elif operation == 'edge':
        processed = edge_canny(image)
    elif operation == 'super_resolve':
        processed = super_resolve_resize(image)
    elif operation == 'inpaint':
        if mask_bytes:
            pil_mask = Image.open(io.BytesIO(mask_bytes)).convert('L')
            mask = np.array(pil_mask)
        else:
            mask = None
        processed = inpaint_telea(image, mask)
    else:
        processed = image

    # Convert processed image from BGR to RGB before saving with PIL
    processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    pil_processed = Image.fromarray(processed_rgb)

    # Save to bytes
    output_bytes = io.BytesIO()
    pil_processed.save(output_bytes, format='PNG')
    output_bytes.seek(0)

    return output_bytes

def process_image(image_path, operation=None, filter_type=None, mask_path=None):
    # Load image from file path
    pil_image = Image.open(image_path).convert('RGB')
    image = np.array(pil_image)[:, :, ::-1]  # Convert RGB to BGR for OpenCV

    mask = None
    if mask_path:
        pil_mask = Image.open(mask_path).convert('L')
        mask = np.array(pil_mask)

    if operation == 'denoise':
        processed = denoise_opencv(image)
    elif operation == 'restore':
        processed = restore_image(image)
    elif operation == 'filter':
        processed = filter_image(image, filter_type)
    elif operation == 'edge':
        processed = edge_canny(image)
    elif operation == 'super_resolve':
        processed = super_resolve_resize(image)
    elif operation == 'inpaint':
        processed = inpaint_telea(image, mask)
    else:
        processed = image

    # Convert processed image from BGR to RGB before saving with PIL
    processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    pil_processed = Image.fromarray(processed_rgb)

    # Save processed image to a new file
    base, ext = os.path.splitext(image_path)
    output_path = f"{base}_processed.png"
    pil_processed.save(output_path)

    return output_path
