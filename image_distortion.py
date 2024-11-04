import cv2
import numpy as np

def add_gaussian_noise(image, mean=0, std=1):

    gauss = np.random.normal(mean, std, image.shape).astype('uint8')
    noisy_image = cv2.add(image, gauss)
    return noisy_image

def apply_gaussian_blur(image, kernel_size=(5, 5), sigma=0):

    blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)
    return blurred_image

def convert_to_grayscale(image):

    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)

def apply_negative_effect(image):

    negative_image = cv2.bitwise_not(image)
    return negative_image

def zoom_image(image, zoom_factor=1.0):

    height, width = image.shape[:2]
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)
    resized_image = cv2.resize(image, (new_width, new_height))

    if zoom_factor > 1.0:
        crop_x = (new_width - width) // 2
        crop_y = (new_height - height) // 2
        zoomed_image = resized_image[crop_y:crop_y + height, crop_x:crop_x + width]
    else:
        pad_x = (width - new_width) // 2
        pad_y = (height - new_height) // 2
        zoomed_image = cv2.copyMakeBorder(resized_image, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return zoomed_image