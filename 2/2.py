import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

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

def cut_object(original_image, mask):
    return cv2.bitwise_and(original_image, original_image, mask=mask)

image_names = ["146.jpg", "istockphoto-1069656514-612x612.jpg", "OIP (2).jpg", "OIP (3).jpg"]

window_size = 15
k_niblack = -0.2
k_sauvola = 0.5
k_christian = 0.3

for image_name in image_names:

    input_image = cv2.imread(image_name)
    if input_image is None:
        print(f"Зображення '{image_name}' не знайдено. Пропускаємо.")
        continue

    # Перетворення в ч/б
    grayscale_image = convert_to_grayscale(input_image)

    plt.imshow(grayscale_image, cmap='gray')
    plt.title(f"Grayscale: {image_name}")
    plt.axis('off')
    plt.show()

    binary_otsu = otsu_binarization(grayscale_image)
    binary_niblack = niblack_binarization(grayscale_image, window_size, k_niblack)
    binary_sauvola = sauvola_binarization(grayscale_image, window_size, k_sauvola)
    binary_christian = christian_binarization(grayscale_image, window_size, k_christian)

    cv2.imwrite(f"Binary_Otsu_{image_name}", binary_otsu)
    cv2.imwrite(f"Binary_Niblack_{image_name}", binary_niblack)
    cv2.imwrite(f"Binary_Sauvola_{image_name}", binary_sauvola)
    cv2.imwrite(f"Binary_Christian_{image_name}", binary_christian)

  
    plt.imshow(binary_otsu, cmap='gray')
    plt.title(f"Binary Otsu: {image_name}")
    plt.axis('off')
    plt.show()

    plt.imshow(binary_niblack, cmap='gray')
    plt.title(f"Binary Niblack: {image_name}")
    plt.axis('off')
    plt.show()

    plt.imshow(binary_sauvola, cmap='gray')
    plt.title(f"Binary Sauvola: {image_name}")
    plt.axis('off')
    plt.show()

    plt.imshow(binary_christian, cmap='gray')
    plt.title(f"Binary Christian: {image_name}")
    plt.axis('off')
    plt.show()


# test_image = np.array([[0,255,0,0,255,0],
#                        [255,255,255,0,255,0],
#                        [0,255,0,0,255,255],
#                        [0,255,255,0,255,255],
#                        [255,0,255,0,0,255],
#                        [0,0,0,0,255,0]])
# test_image = np.zeros((101,101))
# test_image[0, 0] = 255
# test_image[0, 100] = 255
# test_image[100, 100] = 255
# test_image[100, 0] = 255
# test_kernel = np.array([[0,1,0],
#                         [0,0,0],
#                         [0,0,0]])

# Создаем массив 21x21, заполненный нулями

# move_20down_and_10right = np.zeros((41, 41))
# move_20down_and_10right[0, 9] = 1
# print(move_20down_and_10right)
#
# # test_image_shift = shift_20_bottom(shift_10_right(test_image))
# test_image_shift = filter(test_image, move_20down_and_10right)
# cv2.imwrite(f'Images_output/test_image.jpg', test_image)
# cv2.imwrite(f'Images_output/test_image_shift.jpg', test_image_shift)
