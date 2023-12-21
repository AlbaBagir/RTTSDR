# Augmentation.py
import cv2
import numpy as np

def adjust_brightness(image, brightness_range=(-0.3, 0.3)):
    delta = np.random.uniform(brightness_range[0], brightness_range[1])
    augmented_image = cv2.addWeighted(image, 1 + delta, np.zeros_like(image), 0, 0)
    augmented_image = np.clip(augmented_image, 0, 255).astype(np.uint8)
    return augmented_image

def add_noise(image, noise_percentage=0.15):
    noise = np.random.normal(scale=25, size=image.shape).astype(np.uint8)
    mask = np.random.choice([0, 1], size=image.shape[:2], p=[1 - noise_percentage, noise_percentage])
    noisy_image = np.where(mask[:, :, np.newaxis], image + noise, image)
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image
