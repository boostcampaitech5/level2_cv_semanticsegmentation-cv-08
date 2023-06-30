from albumentations import Compose, HorizontalFlip, Normalize, Resize, Rotate, RandomBrightnessContrast
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
from typing import Union
import torchvision.transforms as transform

def get_size(resize: Union[int, list, tuple]):
    if isinstance(resize, (list, tuple)):
        resize = np.random.choice(resize)
    return resize

def base_augmentation(resize, norm=False, mean=0.12397208368416988, std=0.16831689773326278):
    resize = get_size(resize)
    transforms = [Resize(resize, resize, p=1), ToTensorV2(transpose_mask=True)]
    if norm:
        transforms.append(Normalize(mean=mean, std=std, max_pixel_value=1.0, p=1.0))
    return Compose(transforms)

def horizontal_flip(resize, p=0.5, norm=False, mean=0.12397208368416988, std=0.16831689773326278):
    resize = get_size(resize)
    transforms = [HorizontalFlip(p=p), Resize(resize, resize, p=1), ToTensorV2(transpose_mask=True)]
    if norm:
        transforms.append(Normalize(mean=mean, std=std, max_pixel_value=1.0, p=1.0), )
    return Compose(transforms)

def rotate(resize, limit=30, p=0.5, norm=False, mean=0.12397208368416988, std=0.16831689773326278):
    resize = get_size(resize)
    transforms = [Rotate(limit=limit, p=p), Resize(resize, resize, p=1)]
    if norm:
        transforms.append(Normalize(mean=mean, std=std, max_pixel_value=1.0, p=1.0))
    return Compose(transforms)

def rotate_flip(resize, limit=30, p_rotate=0.5, p_flip=0.5, norm=False, mean=0.12397208368416988, std=0.16831689773326278):
    resize = get_size(resize)
    transforms = [Rotate(limit=limit, p=p_rotate), HorizontalFlip(p=p_flip), Resize(resize, resize, p=1), ToTensorV2(transpose_mask=True)]
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

class AdjustContrast:
    """Adjust contrast of an image
    Args:
        resize (int): desired output size
        contrast_factor (float): non-negative number
            0: solid gray
            1: original
            2: increase the contrast by a factor of 2
    """

    def __init__(self, resize, contrast_factor):
        self.contrast_factor = contrast_factor
        self.resize = Resize(resize, resize, p=1)

    def __call__(self, **kwargs):
        image = kwargs["image"]
        return_mask = False
        if "mask" in kwargs:
            return_mask = True
            mask = kwargs["mask"]

        image = transform.ToPILImage()((image * 255).astype(np.uint8))
        image = transform.functional.adjust_contrast(image, self.contrast_factor)
        image = np.asarray(image) / 255.0

        image = self.resize(image=image)

        if return_mask:
            mask = self.resize(image=mask)
            return {"image": image["image"], "mask": mask["image"]}

        return {"image": image["image"]}


def adjust_contrast(resize, contrast_factor):
    resize = get_size(resize)
    transforms = AdjustContrast(resize, contrast_factor)
    return transforms

def brightness_contrast(resize, brightness_limit=[0.1, 0.3], contrast_limit=[0.1, 0.3], p_contrast=0.5, norm=False, mean=0.12397208368416988, std=0.16831689773326278):
    resize = get_size(resize)
    transforms = [RandomBrightnessContrast(brightness_limit=brightness_limit,
                                           contrast_limit=contrast_limit,
                                           p=p_contrast), Resize(resize, resize, p=1)]
    if norm:
        transforms.append(Normalize(mean=mean, std=std, max_pixel_value=1.0, p=1.0))
    return Compose(transforms)