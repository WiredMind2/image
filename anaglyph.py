import numpy as np
from PIL import Image

def load_image(path):
    return np.array(Image.open(path))

def shift_image(image, shift):
    return np.roll(image, shift, axis=1)

def combine_anaglyph(image_left, image_right):
    anaglyph = np.zeros_like(image_left)

    # Combine the red channel of the left image with the blue and green channels of the right image
    anaglyph[:, :, 0] = image_left[:, :, 0]
    anaglyph[:, :, 1] = image_right[:, :, 1] * 0.3 + image_right[:, :, 2] * 0.7
    anaglyph[:, :, 2] = image_right[:, :, 1] * 0.3 + image_right[:, :, 2] * 0.7

    return anaglyph

def save_anaglyph(anaglyph, output_path):
    img = Image.fromarray(anaglyph.astype(np.uint8))
    img.save(output_path)

def main():
    input_left_path = "output/temp1.jpg"
    input_right_path = "output/temp2.jpg"
    output_path = "output/output_anaglyph.jpg"
    shift = 5  # Shift the right image by 80 pixels

    image_left = load_image(input_left_path)
    image_right = shift_image(load_image(input_right_path), shift)
    anaglyph = combine_anaglyph(image_left, image_right)
    save_anaglyph(anaglyph, output_path)

if __name__ == "__main__":
    main()