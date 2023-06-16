from albumentations import Compose, HorizontalFlip, Normalize, Resize, Rotate, RandomBrightnessContrast
import numpy as np
from typing import Union

def get_size(resize: Union[int, list, tuple]):
    if isinstance(resize, (list, tuple)):
        resize = np.random.choice(resize)
    return resize

def base_augmentation(resize, norm=False, mean=0.12397208368416988, std=0.16831689773326278):
    resize = get_size(resize)
    transforms = [Resize(resize, resize, p=1)]
    if norm:
        transforms.append(Normalize(mean=mean, std=std, max_pixel_value=1.0, p=1.0))
    return Compose(transforms)

def horizontal_flip(resize, p=0.5, norm=False, mean=0.12397208368416988, std=0.16831689773326278):
    resize = get_size(resize)
    transforms = [HorizontalFlip(p=p), Resize(resize, resize, p=1)]
    if norm:
        transforms.append(Normalize(mean=mean, std=std, max_pixel_value=1.0, p=1.0))
    return Compose(transforms)

def rotate(resize, limit=30, p=0.5, norm=False, mean=0.12397208368416988, std=0.16831689773326278):
    resize = get_size(resize)
    transforms = [Rotate(limit=limit, p=p), Resize(resize, resize, p=1)]
    if norm:
        transforms.append(Normalize(mean=mean, std=std, max_pixel_value=1.0, p=1.0))
    return Compose(transforms)

def rotate_flip(resize, limit=30, p_rotate=0.5, p_flip=0.5, norm=False, mean=0.12397208368416988, std=0.16831689773326278):
    resize = get_size(resize)
    transforms = [Rotate(limit=limit, p=p_rotate), HorizontalFlip(p=p_flip), Resize(resize, resize, p=1)]
    if norm:
        transforms.append(Normalize(mean=mean, std=std, max_pixel_value=1.0, p=1.0))
    return Compose(transforms)

def brightness_contrast(resize, brightness_limit=[0.1, 0.3], contrast_limit=[0.1, 0.3], p_contrast=0.5, norm=False, mean=0.12397208368416988, std=0.16831689773326278):
    resize = get_size(resize)
    transforms = [RandomBrightnessContrast(brightness_limit=brightness_limit,
                                           contrast_limit=contrast_limit,
                                           p=p_contrast), Resize(resize, resize, p=1)]
    if norm:
        transforms.append(Normalize(mean=mean, std=std, max_pixel_value=1.0, p=1.0))
    return Compose(transforms)