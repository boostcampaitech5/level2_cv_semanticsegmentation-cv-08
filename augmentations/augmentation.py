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
