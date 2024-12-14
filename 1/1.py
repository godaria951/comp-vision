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

# Функція для вирізання об'єкта за допомогою маски
def cut_object(original_image, mask):
    return cv2.bitwise_and(original_image, original_image, mask=mask)

# Завантажте свої зображення в Colab перед запуском
image_names = ["OIP (1).jpg", "OIP.jpg", "завантаження.jpg", "cat (1).jpg"]

# Обробка кожного зображення
for image_name in image_names:
    # Зчитування зображення
    input_image = cv2.imread(image_name)
    if input_image is None:
        print(f"Зображення '{image_name}' не знайдено. Пропускаємо.")
        continue

    # Перетворення в ч/б
    grayscale_image = convert_to_grayscale(input_image)

    # Відображення ч/б зображення
    plt.imshow(grayscale_image, cmap='gray')
    plt.title(f"Grayscale: {image_name}")
    plt.axis('off')
    plt.show()

    # Бінаризація методом Отсу
    binary_mask = otsu_binarization(grayscale_image)

    # Відображення маски
    plt.imshow(binary_mask, cmap='gray')
    plt.title(f"Binary Mask (Otsu): {image_name}")
    plt.axis('off')
    plt.show()

    # Вирізання об'єкта
    object_cut = cut_object(input_image, binary_mask)

    # Збереження результатів
    grayscale_name = f"Grayscale_{image_name}"
    mask_name = f"Mask_{image_name}"
    cut_name = f"Cut_{image_name}"

    cv2.imwrite(grayscale_name, grayscale_image)
    cv2.imwrite(mask_name, binary_mask)
    cv2.imwrite(cut_name, object_cut)

    # Відображення вирізаного об'єкта
    cv2_imshow(object_cut)
