# import os
# import torch
# from torchvision import transforms
# from torch.utils.data import Dataset
# from PIL import Image
# import matplotlib.pyplot as plt

from val.datasets import CustomImageDataset

# # Задание 1
#
# # Создаем пайплайн аугментаций
# augmentations = {
#     'Original': None,
#     'RandomHorizontalFlip': transforms.RandomHorizontalFlip(p=1.0),
#     'RandomCrop': transforms.RandomCrop(size=(180, 180), padding=10),
#     'ColorJitter': transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3),
#     'RandomRotation': transforms.RandomRotation(degrees=45),
#     'RandomGrayscale': transforms.RandomGrayscale(p=1.0),
#     'AllTogether': transforms.Compose([
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.RandomCrop(size=(180, 180), padding=10),
#         transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
#         transforms.RandomRotation(degrees=30),
#         transforms.RandomGrayscale(p=0.2),
#     ])
# }
#
# # Загружаем датасет
# dataset = CustomImageDataset(root_dir='data/train', transform=None)
#
# # Выбираем по одному изображению из каждого класса
# sample_images = []
# sample_labels = []
# for class_idx in range(len(dataset.classes)):
#     for i, label in enumerate(dataset.labels):
#         if label == class_idx:
#             sample_images.append(dataset.images[i])
#             sample_labels.append(label)
#             break
#     if len(sample_images) == 5:
#         break
#
#
# # Функция для визуализации
# def visualize_augmentations(image_path, augmentations):
#     original_image = Image.open(image_path).convert('RGB').resize((224, 224), Image.Resampling.LANCZOS)
#
#     plt.figure(figsize=(15, 10))
#     plt.subplot(2, 4, 1)
#     plt.imshow(original_image)
#     plt.title('Original')
#     plt.axis('off')
#
#     for i, (aug_name, aug) in enumerate(augmentations.items(), 2):
#         if aug_name == 'Original':
#             continue
#
#         plt.subplot(2, 4, i)
#         if aug:
#             augmented_image = aug(original_image)
#         else:
#             augmented_image = original_image
#
#         plt.imshow(augmented_image)
#         plt.title(aug_name)
#         plt.axis('off')
#
#     plt.tight_layout()
#     plt.show()
#
#
# # Визуализируем аугментации
# for img_path, label in zip(sample_images, sample_labels):
#     class_name = dataset.classes[label]
#     print(f"\nАугментации для класса: {class_name}")
#     visualize_augmentations(img_path, augmentations)

# Задание 2
# import os
# import cv2
# import numpy as np
# import random
# import torch
# from matplotlib import pyplot as plt
# from PIL import Image
# from val.extra_augs import *
#
# # Исправленные пути к данным с обработкой для Windows
# data_path = os.path.normpath("data/train/")
# characters = ["Гароу", "Генос", "Сайтама", "Соник", "Татсумаки", "Фубуки"]
#
# # Кэширование списка валидных изображений
# _valid_images_cache = None
#
#
# def get_valid_images():
#     global _valid_images_cache
#     if _valid_images_cache is not None:
#         return _valid_images_cache
#
#     valid_images = []
#     for character in characters:
#         char_path = os.path.join(data_path, character)
#         try:
#             files = [f for f in os.listdir(char_path)
#                      if os.path.isfile(os.path.join(char_path, f)) and
#                      not f.startswith('.') and
#                      f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
#
#             for f in files:
#                 img_path = os.path.join(char_path, f)
#                 valid_images.append(img_path)
#
#         except Exception as e:
#             print(f"Ошибка при сканировании {char_path}: {e}")
#             continue
#
#     if not valid_images:
#         raise Exception("Не найдено ни одного валидного изображения!")
#
#     _valid_images_cache = valid_images
#     return valid_images
#
#
# def load_random_image():
#     max_attempts = 20
#     valid_images = get_valid_images()
#
#     for attempt in range(max_attempts):
#         try:
#             img_path = random.choice(valid_images)
#             print(f"Попытка {attempt + 1}: загружаем {img_path}")
#
#             # Сначала пробуем через PIL, так как он лучше обрабатывает пути и форматы
#             try:
#                 img = Image.open(img_path)
#                 if img.mode != 'RGB':
#                     img = img.convert('RGB')
#                 img = np.array(img)
#                 return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#             except Exception as pil_error:
#                 print(f"PIL не смог загрузить {img_path}: {pil_error}")
#                 # Пробуем OpenCV как запасной вариант
#                 img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
#                 if img is not None:
#                     return img
#
#                 raise Exception("Оба метода загрузки не сработали")
#
#         except Exception as e:
#             print(f"Ошибка при загрузке изображения (попытка {attempt + 1}): {e}")
#             continue
#
#     raise Exception(f"Не удалось загрузить изображение после {max_attempts} попыток")
#
#
# # Остальные функции остаются без изменений
# def random_blur(img):
#     blur_type = random.choice(["gaussian", "median"])
#     if blur_type == "gaussian":
#         ksize = random.choice([3, 5, 7])
#         return cv2.GaussianBlur(img, (ksize, ksize), 0)
#     else:
#         ksize = random.choice([3, 5, 7])
#         return cv2.medianBlur(img, ksize)
#
#
# def random_perspective(img):
#     h, w = img.shape[:2]
#     pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
#     pts2 = np.float32([
#         [random.randint(-50, 50), random.randint(-50, 50)],
#         [w + random.randint(-50, 50), random.randint(-50, 50)],
#         [random.randint(-50, 50), h + random.randint(-50, 50)],
#         [w + random.randint(-50, 50), h + random.randint(-50, 50)]
#     ])
#     M = cv2.getPerspectiveTransform(pts1, pts2)
#     return cv2.warpPerspective(img, M, (w, h))
#
#
# def random_brightness_contrast(img):
#     alpha = random.uniform(0.7, 1.3)
#     beta = random.randint(-50, 50)
#     return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
#
#
# def apply_custom_augmentations(img):
#     aug_img = img.copy()
#     if random.random() > 0.5:
#         aug_img = random_blur(aug_img)
#     if random.random() > 0.5:
#         aug_img = random_perspective(aug_img)
#     if random.random() > 0.5:
#         aug_img = random_brightness_contrast(aug_img)
#     return aug_img
#
#
# def apply_extra_augs(img):
#     try:
#         img_tensor = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32) / 255.0)
#         augmentations = [
#             AddGaussianNoise(mean=0., std=0.1),
#             RandomErasingCustom(p=1.0),
#             CutOut(p=1.0),
#             Solarize(threshold=128),
#             Posterize(bits=4),
#             AutoContrast(p=1.0),
#             ElasticTransform(p=1.0)
#         ]
#         aug = random.choice(augmentations)
#         img_augmented = aug(img_tensor.clone())
#         img_augmented = (img_augmented.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
#         return img_augmented
#     except Exception as e:
#         print(f"Ошибка в apply_extra_augs: {e}")
#         return img
#
#
# def compare_augmentations():
#     try:
#         img = load_random_image()
#         if img is None:
#             raise Exception("Загруженное изображение равно None")
#
#         # Конвертируем BGR в RGB для отображения через matplotlib
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         custom_aug = apply_custom_augmentations(img)
#         custom_aug_rgb = cv2.cvtColor(custom_aug, cv2.COLOR_BGR2RGB)
#         extra_aug = apply_extra_augs(img.copy())
#
#         plt.figure(figsize=(15, 5))
#         plt.subplot(1, 3, 1)
#         plt.imshow(img_rgb)
#         plt.title("Original Image")
#         plt.axis('off')
#
#         plt.subplot(1, 3, 2)
#         plt.imshow(custom_aug_rgb)
#         plt.title("Custom Augmentations")
#         plt.axis('off')
#
#         plt.subplot(1, 3, 3)
#         plt.imshow(extra_aug)
#         plt.title("Extra Augmentations")
#         plt.axis('off')
#
#         plt.tight_layout()
#         plt.show()
#
#     except Exception as e:
#         print(f"Ошибка при сравнении аугментаций: {e}")
#
#
# # Проверим сначала доступность файлов
# print("Проверка доступности путей...")
# for character in characters:
#     char_path = os.path.join(data_path, character)
#     if not os.path.exists(char_path):
#         print(f"Папка не существует: {char_path}")
#     else:
#         print(f"Найдена папка: {char_path}")
#         files = os.listdir(char_path)
#         print(f"Найдено файлов: {len(files)}")
#         if files:
#             test_file = os.path.join(char_path, files[0])
#             print(f"Первый файл: {test_file} - существует: {os.path.exists(test_file)}")
#
# # Запускаем сравнение
# for _ in range(5):
#     compare_augmentations()

# Задание 3
# import os
# import matplotlib.pyplot as plt
# import pandas as pd
# from PIL import Image
# from val.datasets import CustomImageDataset  # Импортируем наш датасет
#
# # Путь к данным
# data_path = os.path.normpath("data/train/")
#
# # Создаём датасет без аугментаций (только для анализа)
# dataset = CustomImageDataset(root_dir=data_path, transform=None, target_size=None)
#
# # Собираем статистику
# stats = []
# for img_path, label in zip(dataset.images, dataset.labels):
#     class_name = dataset.classes[label]
#     with Image.open(img_path) as img:
#         width, height = img.size
#     stats.append({
#         "class": class_name,
#         "width": width,
#         "height": height,
#         "aspect_ratio": width / height
#     })
#
# # Конвертируем в DataFrame
# df = pd.DataFrame(stats)
#
# # 1. Количество изображений по классам
# class_counts = df["class"].value_counts()
# print("\nКоличество изображений в каждом классе:")
# print(class_counts)
#
# # 2. Анализ размеров
# print("\nАнализ размеров изображений:")
# print(f"• Минимальный размер: {df[['width', 'height']].min().values} px (ширина, высота)")
# print(f"• Максимальный размер: {df[['width', 'height']].max().values} px")
# print(f"• Средний размер: {df[['width', 'height']].mean().round(1).values} px")
#
# # 3. Визуализация
# plt.figure(figsize=(16, 6))
#
# # Гистограмма по классам
# plt.subplot(1, 3, 1)
# class_counts.plot(kind="bar", color="skyblue")
# plt.title("Распределение по классам", pad=20)
# plt.xlabel("Класс")
# plt.ylabel("Количество изображений")
# plt.xticks(rotation=45)
#
# # Scatter plot размеров
# plt.subplot(1, 3, 2)
# plt.scatter(df["width"], df["height"], alpha=0.5, color="green")
# plt.title("Соотношение ширины и высоты", pad=20)
# plt.xlabel("Ширина (px)")
# plt.ylabel("Высота (px)")
#
# # Гистограмма соотношения сторон
# plt.subplot(1, 3, 3)
# plt.hist(df["aspect_ratio"], bins=20, color="orange", edgecolor="black")
# plt.title("Распределение соотношения сторон", pad=20)
# plt.xlabel("width / height")
# plt.ylabel("Частота")
#
# plt.tight_layout()
# plt.show()

# # Задание 4
#
# import cv2
# import numpy as np
# import random
# from typing import Dict, Callable, List
# from PIL import Image
#
#
# class AugmentationPipeline:
#     def __init__(self):
#         self.augmentations: Dict[str, Callable] = {}
#
#     def add_augmentation(self, name: str, aug: Callable) -> None:
#         """Добавляет аугментацию в пайплайн"""
#         self.augmentations[name] = aug
#
#     def remove_augmentation(self, name: str) -> None:
#         """Удаляет аугментацию по имени"""
#         self.augmentations.pop(name, None)
#
#     def apply(self, image: np.ndarray) -> np.ndarray:
#         """Применяет все аугментации последовательно"""
#         aug_image = image.copy()
#         for aug in self.augmentations.values():
#             aug_image = aug(aug_image)
#         return aug_image
#
#     def get_augmentations(self) -> List[str]:
#         """Возвращает список имен аугментаций"""
#         return list(self.augmentations.keys())
#
#
# # ===== Базовые аугментации =====
# def random_rotate(img: np.ndarray) -> np.ndarray:
#     angle = random.uniform(-15, 15)
#     h, w = img.shape[:2]
#     M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
#     return cv2.warpAffine(img, M, (w, h))
#
#
# def random_flip(img: np.ndarray) -> np.ndarray:
#     if random.random() > 0.5:
#         return cv2.flip(img, 1)  # Горизонтальный флип
#     return img
#
#
# def color_jitter(img: np.ndarray) -> np.ndarray:
#     img = img.astype(np.float32)
#     img += np.random.uniform(-20, 20, 3)  # Яркость
#     img = np.clip(img, 0, 255)
#     return img.astype(np.uint8)
#
#
# def gaussian_blur(img: np.ndarray) -> np.ndarray:
#     return cv2.GaussianBlur(img, (5, 5), 0)
#
#
# def heavy_distortion(img: np.ndarray) -> np.ndarray:
#     # Комбинация нескольких аугментаций
#     img = random_rotate(img)
#     img = color_jitter(img)
#     img = cv2.addWeighted(img, 0.7, np.zeros_like(img), 0, 30)  # Контраст
#     return img
#
#
# # ===== Конфигурации =====
# def create_light_pipeline() -> AugmentationPipeline:
#     pipeline = AugmentationPipeline()
#     pipeline.add_augmentation("flip", random_flip)
#     pipeline.add_augmentation("color_jitter", color_jitter)
#     return pipeline
#
#
# def create_medium_pipeline() -> AugmentationPipeline:
#     pipeline = AugmentationPipeline()
#     pipeline.add_augmentation("rotate", random_rotate)
#     pipeline.add_augmentation("flip", random_flip)
#     pipeline.add_augmentation("blur", gaussian_blur)
#     return pipeline
#
#
# def create_heavy_pipeline() -> AugmentationPipeline:
#     pipeline = AugmentationPipeline()
#     pipeline.add_augmentation("heavy_distortion", heavy_distortion)
#     pipeline.add_augmentation("rotate", random_rotate)
#     pipeline.add_augmentation("blur", gaussian_blur)
#     return pipeline
#
#
# # ===== Пример использования =====
# if __name__ == "__main__":
#     # Загрузка тестового изображения
#     image = np.array(Image.open("data/train/Сайтама/05eaf704e653c1cedad80dedc0e30824.jpg"))
#
#     # Создаем пайплайны
#     light = create_light_pipeline()
#     medium = create_medium_pipeline()
#     heavy = create_heavy_pipeline()
#
#     # Применяем аугментации
#     light_aug = light.apply(image)
#     medium_aug = medium.apply(image)
#     heavy_aug = heavy.apply(image)
#
#     # Визуализация
#     import matplotlib.pyplot as plt
#
#     plt.figure(figsize=(15, 5))
#
#     plt.subplot(1, 4, 1)
#     plt.imshow(image)
#     plt.title("Original")
#
#     plt.subplot(1, 4, 2)
#     plt.imshow(light_aug)
#     plt.title(f"Light: {light.get_augmentations()}")
#
#     plt.subplot(1, 4, 3)
#     plt.imshow(medium_aug)
#     plt.title(f"Medium: {medium.get_augmentations()}")
#
#     plt.subplot(1, 4, 4)
#     plt.imshow(heavy_aug)
#     plt.title(f"Heavy: {heavy.get_augmentations()}")
#
#     plt.tight_layout()
#     plt.show()
#
#     # Сохранение результатов
#     Image.fromarray(light_aug).save("light_aug.jpg")
#     Image.fromarray(medium_aug).save("medium_aug.jpg")
#     Image.fromarray(heavy_aug).save("heavy_aug.jpg")
#
# #

# Задание 6
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
from val.datasets import CustomImageDataset
import matplotlib.pyplot as plt
from tqdm import tqdm


def main():
    # Конфигурация
    BATCH_SIZE = 32
    EPOCHS = 10
    LR = 1e-3
    IMAGE_SIZE = 224

    # 1. Подготовка данных
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CustomImageDataset('data/train', transform=transform)
    val_dataset = CustomImageDataset('data/test', transform=transform)

    # Проверка, что валидационный датасет не пуст
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty. Please check the path 'val' and ensure it contains images.")

    # Убираем num_workers для Windows или ставим 0
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0)

    # 2. Подготовка модели
    def get_model(num_classes):
        model = models.resnet18(weights='IMAGENET1K_V1')

        # Замораживаем все слои кроме последнего
        for param in model.parameters():
            param.requires_grad = False

        # Заменяем последний слой
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    model = get_model(len(train_dataset.get_class_names()))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 3. Обучение
    optimizer = optim.Adam(model.fc.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    accuracies = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{EPOCHS}')

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        # Валидация (только если есть данные)
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        if len(val_loader) > 0:  # Проверка, что есть данные для валидации
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            # Сохраняем метрики
            epoch_train_loss = running_loss / len(train_loader)
            epoch_val_loss = val_loss / len(val_loader)
            epoch_acc = correct / total

            train_losses.append(epoch_train_loss)
            val_losses.append(epoch_val_loss)
            accuracies.append(epoch_acc)

            print(f'Epoch {epoch + 1}: '
                  f'Train Loss: {epoch_train_loss:.4f}, '
                  f'Val Loss: {epoch_val_loss:.4f}, '
                  f'Accuracy: {epoch_acc:.4f}')
        else:
            # Если нет валидационных данных, сохраняем только train loss
            epoch_train_loss = running_loss / len(train_loader)
            train_losses.append(epoch_train_loss)
            print(f'Epoch {epoch + 1}: Train Loss: {epoch_train_loss:.4f}')

    # 4. Визуализация (только если есть валидационные данные)
    if len(val_loader) > 0:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')

        plt.subplot(1, 2, 2)
        plt.plot(accuracies, label='Accuracy', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Validation Accuracy')

        plt.tight_layout()
        plt.savefig('training_metrics.png')
        plt.show()
    else:
        # Визуализация только train loss, если нет валидационных данных
        plt.figure(figsize=(6, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training Loss')
        plt.tight_layout()
        plt.savefig('training_loss.png')
        plt.show()

    # 5. Сохранение модели
    torch.save(model.state_dict(), 'fine_tuned_resnet18.pth')


if __name__ == '__main__':
    main()