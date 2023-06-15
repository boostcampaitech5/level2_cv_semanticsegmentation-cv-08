import numpy as np
import torchvision.transforms as transform
from albumentations import Compose, HorizontalFlip, Normalize, Resize, Rotate


def base_augmentation(resize, norm=False, mean=0.12397208368416988, std=0.16831689773326278):
    transforms = [Resize(resize, resize, p=1)]
    if norm:
        transforms.append(Normalize(mean=mean, std=std, max_pixel_value=1.0, p=1.0))
    return Compose(transforms)


def horizontal_flip(resize, norm=False, mean=0.12397208368416988, std=0.16831689773326278):
    transforms = [HorizontalFlip(p=0.5), Resize(resize, resize, p=1)]
    if norm:
        transforms.append(Normalize(mean=mean, std=std, max_pixel_value=1.0, p=1.0))
    return Compose(transforms)


def rotate(resize, limit=30, norm=False, mean=0.12397208368416988, std=0.16831689773326278):
    transforms = [Rotate(limit=limit), Resize(resize, resize, p=1)]
    if norm:
        transforms.append(Normalize(mean=mean, std=std, max_pixel_value=1.0, p=1.0))
    return Compose(transforms)


def rotate_flip(resize, limit=30, norm=False, mean=0.12397208368416988, std=0.16831689773326278):
    transforms = [Rotate(limit=limit), HorizontalFlip(p=0.5), Resize(resize, resize, p=1)]
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
        image, mask = kwargs["image"], kwargs["mask"]

        image = transform.ToPILImage()((image * 255).astype(np.uint8))
        image = transform.functional.adjust_contrast(image, self.contrast_factor)
        image = np.asarray(image) / 255.0

        image = self.resize(image=image)
        mask = self.resize(image=mask)
        return {"image": image["image"], "mask": mask["image"]}


def adjust_contrast(resize, contrast_factor):
    transforms = AdjustContrast(resize, contrast_factor)
    return transforms
