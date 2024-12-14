import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

# Функція для конвертації зображення в ч/б варіант
def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Функція для бінаризації методом Отсу
def otsu_binarization(image):
    _, binary_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_mask

# Реалізація методу Ніблека
def niblack_binarization(image, window_size, k):
    mean = cv2.blur(image.astype(np.float32), (window_size, window_size))
    stddev = cv2.blur((image.astype(np.float32) - mean) ** 2, (window_size, window_size)) ** 0.5
    threshold = mean + k * stddev
    binary_mask = (image > threshold).astype(np.uint8) * 255
    return binary_mask

# Реалізація методу Саувола
def sauvola_binarization(image, window_size, k, R=128):
    mean = cv2.blur(image.astype(np.float32), (window_size, window_size))
    stddev = cv2.blur((image.astype(np.float32) - mean) ** 2, (window_size, window_size)) ** 0.5
    threshold = mean * (1 + k * (stddev / R - 1))
    binary_mask = (image > threshold).astype(np.uint8) * 255
    return binary_mask

# Реалізація методу Крістіана
def christian_binarization(image, window_size, k):
    mean = cv2.blur(image.astype(np.float32), (window_size, window_size))
    stddev = cv2.blur((image.astype(np.float32) - mean) ** 2, (window_size, window_size)) ** 0.5
    threshold = mean - k * stddev
    binary_mask = (image > threshold).astype(np.uint8) * 255
    return binary_mask

def apply_filter(image, kernel):
    return cv2.filter2D(image, -1, kernel)

def shift_filter(dx, dy):
    kernel = np.zeros((3, 3), np.float32)
    kernel[1 + dy, 1 + dx] = 1
    return kernel

def inversion_filter():
    return -1 * np.eye(3, dtype=np.float32) + 1

def gaussian_filter(size, sigma):
    return cv2.getGaussianKernel(size, sigma) @ cv2.getGaussianKernel(size, sigma).T

def motion_blur_filter(size):
    kernel = np.eye(size, dtype=np.float32) / size
    return kernel

def sharpen_filter():
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    return kernel

def sobel_filter():
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    return kernel

def edge_filter():
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], np.float32)
    return kernel

def custom_filter():
    kernel = np.array([[1, -1, 1], [-1, 4, -1], [1, -1, 1]], np.float32)
    return kernel

image_names = ["cat (1).jpg", "146.jpg"]

for image_name in image_names:
   
    input_image = cv2.imread(image_name)
    if input_image is None:
        print(f"Зображення '{image_name}' не знайдено. Пропускаємо.")
        continue

    grayscale_image = convert_to_grayscale(input_image)

    plt.imshow(grayscale_image, cmap='gray')
    plt.title(f"Grayscale: {image_name}")
    plt.axis('off')
    plt.show()

    filters = {
        "Shift": shift_filter(1, 1),
        "Inversion": inversion_filter(),
        "Gaussian": gaussian_filter(11, 1.5),
        "Motion Blur": motion_blur_filter(7),
        "Sharpen": sharpen_filter(),
        "Sobel": sobel_filter(),
        "Edge": edge_filter(),
        "Custom": custom_filter()
    }

    for filter_name, kernel in filters.items():
        filtered_image = apply_filter(grayscale_image, kernel)

        plt.imshow(filtered_image, cmap='gray')
        plt.title(f"{filter_name} Filter: {image_name}")
        plt.axis('off')
        plt.show()
