import albumentations as A
import albumentations.pytorch


def getTransforms(a_opt, t_opt, resizeflag):
    translate_per = dict()
    translate_per['x'] = (-0.1, 0.1)
    translate_per['y'] = (-0.1, 0)

    augment1_need1 = A.Affine(scale=None, translate_percent=None, rotate=(-10, 10), p=1.0)
    augment1_need2 = A.Affine(scale=None, translate_percent=translate_per, rotate=(-10, 10), p=1.0)
    augment2_option = A.GaussNoise(per_channel=False, p=1.0)
    augment3_option1 = A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0)
    augment3_option2 = A.ColorJitter(brightness=0.1, contrast=0.1)

    transform1_option = A.ToSepia(p=1.0)
    transform2_option1 = A.CLAHE(p=1.0)
    transform2_option2 = A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0)
    transform3_need1 = A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
    transform3_need2 = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    augment4_always = A.HorizontalFlip(p=0.5)
    transform4_always = A.pytorch.transforms.ToTensorV2()

    resizing_transform = A.Resize(224, 224)

    if resizeflag:
        prefix = []
        suffix = [transform4_always]
    else:
        prefix = [A.CenterCrop(width=900, height=900, p=1.0)]
        suffix = [resizing_transform, transform4_always]

    augments = [augment4_always]
    transforms = []

    if a_opt[0] == '0':
        augments.append(augment1_need1)
    else:
        augments.append(augment1_need2)

    if a_opt[1] == '1':
        augments.append(augment2_option)

    if a_opt[2] == '1':
        augments.append(augment3_option1)
    elif a_opt[2] == '2':
        augments.append(augment3_option2)

    if t_opt[0] == '1':
        transforms.append(transform1_option)

    if t_opt[1] == '1':
        transforms.append(transform2_option1)
    elif t_opt[1] == '2':
        transforms.append(transform2_option2)

    if t_opt[2] == '0':
        transforms.append(transform3_need1)
    else:
        transforms.append(transform3_need2)

    train_transform = A.Compose(
        prefix + augments + transforms + suffix
    )
    test_transform = A.Compose(
        prefix + transforms + suffix
    )

    return train_transform, test_transform
