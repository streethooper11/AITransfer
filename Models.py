import torchvision
from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class AlexNet:
    def __init__(self, num_classes=2):
        self.name = 'AlexNet'
        self.model = torchvision.models.alexnet(weights='DEFAULT')
        self.model.classifier[-1] = nn.Linear(in_features=4096, out_features=num_classes, bias=True)
        self.transform = transforms.Compose(
            [
                transforms.Resize(256, InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])


class MobileNetV3L:
    def __init__(self, num_classes=2):
        self.name = 'MobileNetV3L'
        self.model = torchvision.models.mobilenet_v3_large(weights='DEFAULT')
        self.model.classifier[-1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)
        self.transform = transforms.Compose(
            [
                transforms.Resize(232, InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])


class EfficientNetB0:
    def __init__(self, num_classes=2):
        self.name = 'EfficientNetB0'
        self.model = torchvision.models.efficientnet_b0(weights='DEFAULT')
        self.model.classifier[-1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)
        self.transform = transforms.Compose(
            [
                transforms.Resize(256, InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])


class MaxVitT:
    def __init__(self, num_classes=2):
        self.name = 'MaxvitT'
        self.model = torchvision.models.maxvit_t(weights='DEFAULT')
        self.model.classifier[-1] = nn.Linear(in_features=512, out_features=num_classes, bias=False)
        self.transform = transforms.Compose(
            [
                transforms.Resize(224, InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])


class ResNet18:
    def __init__(self, num_classes=2):
        self.name = 'ResNet18'
        self.model = torchvision.models.resnet18(weights='DEFAULT')
        self.model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        self.transform = transforms.Compose(
            [
                transforms.Resize(256, InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])


class ResNet50:
    def __init__(self, num_classes=2):
        self.name = 'ResNet50'
        self.model = torchvision.models.resnet50(weights='DEFAULT')
        self.model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        self.transform = transforms.Compose(
            [
                transforms.Resize(232, InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])


class RegNetY400MF:
    def __init__(self, num_classes=2):
        self.name = 'RegNetY400MF'
        self.model = torchvision.models.regnet_y_400mf(weights='DEFAULT')
        self.model.fc = nn.Linear(in_features=440, out_features=num_classes, bias=True)
        self.transform = transforms.Compose(
            [
                transforms.Resize(232, InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])


class MNasNet05:
    def __init__(self, num_classes=2):
        self.name = 'MNasNet05'
        self.model = torchvision.models.mnasnet0_5(weights='DEFAULT')
        self.model.fc = nn.Linear(in_features=1280, out_features=num_classes, bias=True)
        self.transform = transforms.Compose(
            [
                transforms.Resize(256, InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])


class InceptionV3:
    def __init__(self, num_classes=2):
        self.name = 'InceptionV3'
        self.model = torchvision.models.inception_v3(weights='DEFAULT')
        self.model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        self.transform = transforms.Compose(
            [
                transforms.Resize(342, InterpolationMode.BILINEAR),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
