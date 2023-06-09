from albumentations.pytorch.transforms import ToTensorV2
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    IAAPerspective,
    ShiftScaleRotate,
    CLAHE,
    RandomRotate90,
    Transpose,
    Blur,
    OpticalDistortion,
    GridDistortion,
    HueSaturationValue,
    IAAAdditiveGaussianNoise,
    GaussNoise,
    MotionBlur,
    MedianBlur,
    IAAPiecewiseAffine,
    RandomResizedCrop,
    IAASharpen,
    IAAEmboss,
    RandomBrightnessContrast,
    Flip,
    OneOf,
    Compose,
    Normalize,
    Cutout,
    ToGray,
    CoarseDropout,
    CenterCrop,
    Resize,
    Rotate,
)

def base_augmentation(resize, norm=True, mean=0.13189, std=0.17733):
    transforms = [Resize(resize, resize, p=1)]
    if norm:
        transforms.append(Normalize(mean=mean, std=std, max_pixel_value=1., p=1.0))
    return Compose(transforms)

def horizontal_flip(resize, norm=True, mean=0.13189, std=0.17733):
    transforms = [HorizontalFlip(p=0.5), Resize(resize, resize, p=1)]
    if norm:
        transforms.append(Normalize(mean=mean, std=std, max_pixel_value=1., p=1.0))
    return Compose(transforms)

def rotate(resize, limit=30, norm=True, mean=0.13189, std=0.17733):
    transforms = [Rotate(limit=limit),
                  Resize(resize, resize, p=1)]
    if norm:
        transforms.append(Normalize(mean=mean, std=std, max_pixel_value=1., p=1.0))
    return Compose(transforms)

def rotate_flip(resize, limit=30, norm=True, mean=0.13189, std=0.17733):
    transforms = [Rotate(limit=limit),
                  HorizontalFlip(p=0.5),
                  Resize(resize, resize, p=1)]
    if norm:
        transforms.append(Normalize(mean=mean, std=std, max_pixel_value=1., p=1.0))
    return Compose(transforms)

