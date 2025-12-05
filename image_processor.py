import os
import cv2
import numpy as np
from typing import Optional

def enhance_image_for_ocr(
    image_path: str,
    scale_factor: float = 1.5,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Loads an image and applies pre-processing techniques to enhance text
    contrast and clarity for OCR.
    """
    # 1. Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not open or find the image at: {image_path}")

    # 2. Rescale/Upscale (Improves DPI and small text)
    h, w = img.shape[:2]
    img = cv2.resize(img, (int(w * scale_factor), int(h * scale_factor)),
                     interpolation=cv2.INTER_LINEAR)

    # 3. Grayscale Conversion
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 4. Noise Reduction
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    # 5. Adaptive Thresholding (Binarization)
    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )

    # 6. Save the result (optional)
    if save_path:
        cv2.imwrite(save_path, binary)
        print(f"✅ Enhanced image saved to {save_path}")

    return binary


def process_images_in_folder(input_folder: str, output_folder: str, scale_factor: float = 1.5):
    """
    Processes all images in a folder and saves enhanced versions in another folder.
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Supported image extensions
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    # List all image files in input folder
    images = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]
    if not images:
        print("⚠️ No image files found in the input folder.")
        return

    print(f"Found {len(images)} image(s) in '{input_folder}'")

    # Process each image
    for idx, filename in enumerate(images, start=1):
        input_path = os.path.join(input_folder, filename)
        name, _ = os.path.splitext(filename)
        output_path = os.path.join(output_folder, f"{name}_enhanced.png")

        try:
            enhance_image_for_ocr(input_path, scale_factor=scale_factor, save_path=output_path)
            print(f"[{idx}/{len(images)}] Processed: {filename}")
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

    print(f"\n✅ All images processed. Enhanced files saved in: {output_folder}")


if __name__ == "__main__":
    # 💡 EDIT THESE PATHS
    input_folder = r"C:\\Projects\\TVS\\Input_images"
    output_folder = r"C:\\Projects\\TVS\\preprocessed_images"

    process_images_in_folder(input_folder, output_folder, scale_factor=1.5)
