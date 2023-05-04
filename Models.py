import torchvision
from torch import nn, optim
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class GenericModel:
    def __init__(self, model):
        self.model = model
        self.optimwithstr = [
#            (optim.Adam(model.parameters(), lr=0.000001), '_Adam_', '0.000001'),
            (optim.Adagrad(model.parameters(), lr=0.000002), '_Adagrad_', '0.000002'),
            (optim.Adamax(model.parameters(), lr=0.000002), '_Adamax_', '0.000002'),
            (optim.RMSprop(model.parameters(), lr=0.000002, momentum=0.9), '_RMSProp_', '0.000002'),
            (optim.SGD(model.parameters(), lr=0.000002, momentum=0.9), '_SGD_', '0.000002'),
        ]


class AlexNet(GenericModel):
    def __init__(self, num_classes=2):
        self.name = 'AlexNet'
        super().__init__(torchvision.models.alexnet(weights='DEFAULT'))
        self.model.classifier[-1] = nn.Linear(in_features=4096, out_features=num_classes, bias=True)
        self.transform = transforms.Compose(
            [
                transforms.Resize(256, InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])


class EfficientNetB0(GenericModel):
    def __init__(self, num_classes=2):
        self.name = 'EfficientNetB0'
        super().__init__(torchvision.models.efficientnet_b0(weights='DEFAULT'))
        self.model.classifier[-1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)
        self.transform = transforms.Compose(
            [
                transforms.Resize(256, InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])


class MaxVitT(GenericModel):
    def __init__(self, num_classes=2):
        self.name = 'MaxVitT'
        super().__init__(torchvision.models.maxvit_t(weights='DEFAULT'))
        self.model.classifier[-1] = nn.Linear(in_features=512, out_features=num_classes, bias=False)
        self.transform = transforms.Compose(
            [
                transforms.Resize(224, InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])


class MNasNet05(GenericModel):
    def __init__(self, num_classes=2):
        self.name = 'MNasNet05'
        super().__init__(torchvision.models.mnasnet0_5(weights='DEFAULT'))
        self.model.classifier[-1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)
        self.transform = transforms.Compose(
            [
                transforms.Resize(256, InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])


class MobileNetV3L(GenericModel):
    def __init__(self, num_classes=2):
        self.name = 'MobileNetV3L'
        super().__init__(torchvision.models.mobilenet_v3_large(weights='DEFAULT'))
        self.model.classifier[-1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)
        self.transform = transforms.Compose(
            [
                transforms.Resize(232, InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])


class RegNetY400MF(GenericModel):
    def __init__(self, num_classes=2):
        self.name = 'RegNetY400MF'
        super().__init__(torchvision.models.regnet_y_400mf(weights='DEFAULT'))
        self.model.fc = nn.Linear(in_features=440, out_features=num_classes, bias=True)
        self.transform = transforms.Compose(
            [
                transforms.Resize(232, InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])


class ResNet18(GenericModel):
    def __init__(self, num_classes=2):
        self.name = 'ResNet18'
        super().__init__(torchvision.models.resnet18(weights='DEFAULT'))
        self.model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        self.transform = transforms.Compose(
            [
                transforms.Resize(256, InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])


class ResNet50(GenericModel):
    def __init__(self, num_classes=2):
        self.name = 'ResNet50'
        super().__init__(torchvision.models.resnet50(weights='DEFAULT'))
        self.model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        self.transform = transforms.Compose(
            [
                transforms.Resize(232, InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])


class ResNext50(GenericModel):
    def __init__(self, num_classes=2):
        self.name = 'ResNext50'
        super().__init__(torchvision.models.resnext50_32x4d(weights='DEFAULT'))
        self.model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        self.transform = transforms.Compose(
            [
                transforms.Resize(232, InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])


class ShuffleNetV205(GenericModel):
    def __init__(self, num_classes=2):
        self.name = 'ShuffleNetV205'
        super().__init__(torchvision.models.shufflenet_v2_x0_5(weights='DEFAULT'))
        self.model.fc = nn.Linear(in_features=1024, out_features=num_classes, bias=True)
        self.transform = transforms.Compose(
            [
                transforms.Resize(256, InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])


class SqueezeNet10(GenericModel):
    def __init__(self, num_classes=2):
        self.name = 'SqueezeNet10'
        super().__init__(torchvision.models.squeezenet1_0(weights='DEFAULT'))
        self.model.classifier[-3] = nn.Conv2d(512, num_classes, kernel_size=1)
        self.transform = transforms.Compose(
            [
                transforms.Resize(256, InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])


class VitB16(GenericModel):
    def __init__(self, num_classes=2):
        self.name = 'VitB16'
        super().__init__(torchvision.models.vit_b_16(weights='DEFAULT'))
        self.model.heads[-1] = nn.Linear(in_features=768, out_features=num_classes, bias=True)
        self.transform = transforms.Compose(
            [
                transforms.Resize(256, InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])


class Vgg11(GenericModel):
    def __init__(self, num_classes=2):
        self.name = 'Vgg11'
        super().__init__(torchvision.models.vgg11(weights='DEFAULT'))
        self.model.classifier[-1] = nn.Linear(in_features=4096, out_features=num_classes, bias=True)
        self.transform = transforms.Compose(
            [
                transforms.Resize(256, InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
