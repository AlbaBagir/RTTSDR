# preprocessing.py

import os
import cv2
import numpy as np
from Augmentation import adjust_brightness, add_noise

def resize_image(image, target_size=(640, 640)):
    return cv2.resize(image, target_size)

def read_annotation(annotation_file):
    with open(annotation_file, 'r') as f:
        lines = f.readlines()
    annotations = [tuple(map(float, line.strip().split())) for line in lines]
    return annotations

def preprocess_image(image_path):
    image = cv2.imread(image_path)

    resized_image = resize_image(image)

    annotation_file = os.path.splitext(image_path)[0] + '.txt'
    annotations = read_annotation(annotation_file)

    augmentation_functions = [adjust_brightness, add_noise]
    selected_augmentation = np.random.choice(augmentation_functions)
    augmented_image = selected_augmentation(resized_image)

    return augmented_image, annotations

def crop_image(image_path, target_size):
    image = cv2.imread(image_path)
    noise = np.random.normal(scale=25, size=image.shape).astype(np.uint8)
    mask = np.random.choice([0, 1], size=image.shape[:2], p=[1 - noise])
    noisy_image = np.where(mask[:, :, np.newaxis], image + noise, image)
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    cropped_image = image(target_size)* mask

    return cropped_image
