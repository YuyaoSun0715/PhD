import os
from PIL import Image, ImageOps

# Minimum dimensions for the images
MIN_SIZE = 84


def pad_image(image: Image.Image, min_size: int = MIN_SIZE) -> Image.Image:
    """
    Pads the image equally on all sides if its width or height is smaller than min_size.
    Padding is done with black (zero) pixels.
    """
    width, height = image.size
    pad_width = max(0, min_size - width)
    pad_height = max(0, min_size - height)

    # Calculate padding amounts for each side
    left = pad_width // 2
    right = pad_width - left
    top = pad_height // 2
    bottom = pad_height - top

    padded_image = ImageOps.expand(image, border=(left, top, right, bottom), fill=0)
    return padded_image


def process_images_in_folder(input_folder: str, output_folder: str):
    """
    Recursively processes image files in input_folder. For any image with dimensions smaller
    than MIN_SIZE, pads it and saves it to a corresponding folder structure in output_folder.
    Images that already meet the size requirement are simply copied.
    """
    for root, _, files in os.walk(input_folder):
        # Determine the relative path to preserve folder structure in the output
        rel_path = os.path.relpath(root, input_folder)
        output_dir = os.path.join(output_folder, rel_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for filename in files:
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                input_path = os.path.join(root, filename)
                output_path = os.path.join(output_dir, filename)
                try:
                    with Image.open(input_path) as img:
                        width, height = img.size
                        if width < MIN_SIZE or height < MIN_SIZE:
                            padded_img = pad_image(img, MIN_SIZE)
                            padded_img.save(output_path)
                            print(f"Padded '{input_path}' from ({width}x{height}) to {padded_img.size}")
                        else:
                            # Copy image as is if it meets the size requirements
                            img.save(output_path)
                            print(f"Copied '{input_path}' as it meets the size requirements: {img.size}")
                except Exception as e:
                    print(f"Error processing {input_path}: {e}")


if __name__ == "__main__":
    # Set the input and output folder paths based on your project structure.
    # Input: data/test
    # Output: data/padded_test
    current_dir = os.getcwd()
    input_folder = os.path.join(current_dir, "data", "test")
    output_folder = os.path.join(current_dir, "data", "padded_test")

    process_images_in_folder(input_folder, output_folder)

