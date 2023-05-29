import torchvision
from torch import nn, optim


class GenericModel:
    def __init__(self, name, model):
        self.name = name
        self.model = model(weights='DEFAULT')
        self.opt = None
        self.params_update = []
        self.define_grads_and_last_layer(feature_extract=True)


    def defineOpt(self, opt):
        if (opt[0] is optim.SGD) or (opt[0] is optim.RMSprop):
            self.opt = (opt[0](self.params_update, lr=float(opt[2]), weight_decay=opt[3], momentum=0.9),
                        opt[1], opt[2], opt[3], opt[4])
        else:
            self.opt = (opt[0](self.params_update, lr=float(opt[2]), weight_decay=opt[3]),
                        opt[1], opt[2], opt[3], opt[4])


    def define_grads_and_last_layer(self, feature_extract):
        if feature_extract:
            for param in self.model.parameters():
                param.requires_grad = False

        else:
            for param in self.model.parameters():
                param.requires_grad = True

    def updateparams(self):
        self.params_update.clear()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.params_update.append(param)


class AlexNet(GenericModel):
    def __init__(self, num_classes, device, opt):
        super().__init__('AlexNet', torchvision.models.alexnet)
        self.model.classifier[-1] = nn.Linear(in_features=4096, out_features=num_classes, bias=True)
        self.updateparams()
        self.model.to(device)
        self.defineOpt(opt)
        # self.defineTransform(256, 224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class DenseNet121(GenericModel):
    def __init__(self, num_classes, device, opt):
        super().__init__('DenseNet121', torchvision.models.densenet121)
        self.model.classifier = nn.Linear(in_features=1024, out_features=num_classes, bias=True)
        self.updateparams()
        self.model.to(device)
        self.defineOpt(opt)
        # self.defineTransform(256, 224, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])


class EfficientNetB0(GenericModel):
    def __init__(self, num_classes, device, opt):
        super().__init__('EfficientNetB0', torchvision.models.efficientnet_b0)
        self.model.classifier[-1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)
        self.updateparams()
        self.model.to(device)
        self.defineOpt(opt)
        # self.defineTransform(256, 224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class MaxVitT(GenericModel):
    def __init__(self, num_classes, device, opt):
        super().__init__('MaxVitT', torchvision.models.maxvit_t)
        self.model.classifier[-1] = nn.Linear(in_features=512, out_features=num_classes, bias=False)
        self.updateparams()
        self.model.to(device)
        self.defineOpt(opt)
        # self.defineTransform(224, 224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class MNasNet05(GenericModel):
    def __init__(self, num_classes, device, opt):
        super().__init__('MNasNet05', torchvision.models.mnasnet0_5)
        self.model.classifier[-1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)
        self.updateparams()
        self.model.to(device)
        self.defineOpt(opt)
        # self.defineTransform(256, 224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class MobileNetV2(GenericModel):
    def __init__(self, num_classes, device, opt):
        super().__init__('MobileNetV2', torchvision.models.mobilenet_v2)
        self.model.classifier[-1] = nn.Linear(in_features=1280, out_features=num_classes)
        self.updateparams()
        self.model.to(device)
        self.defineOpt(opt)
        # self.defineTransform(232, 224, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])


class MobileNetV3L(GenericModel):
    def __init__(self, num_classes, device, opt):
        super().__init__('MobileNetV3L', torchvision.models.mobilenet_v3_large)
        self.model.classifier[-1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)
        self.updateparams()
        self.model.to(device)
        self.defineOpt(opt)
        # self.defineTransform(232, 224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class RegNetY400MF(GenericModel):
    def __init__(self, num_classes, device, opt):
        super().__init__('RegNetY400MF', torchvision.models.regnet_y_400mf)
        self.model.fc = nn.Linear(in_features=440, out_features=num_classes, bias=True)
        self.updateparams()
        self.model.to(device)
        self.defineOpt(opt)
        # self.defineTransform(232, 224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class ResNet18(GenericModel):
    def __init__(self, num_classes, device, opt):
        super().__init__('ResNet18', torchvision.models.resnet18)
        self.model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        self.updateparams()
        self.model.to(device)
        self.defineOpt(opt)
        # self.defineTransform(256, 224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class ResNet50(GenericModel):
    def __init__(self, num_classes, device, opt):
        super().__init__('ResNet50', torchvision.models.resnet50)
        self.model.fc = nn.Linear(in_features=2048, out_features=num_classes)
        self.updateparams()
        self.model.to(device)
        self.defineOpt(opt)
        # self.defineTransform(232, 224, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])


class ResNext50(GenericModel):
    def __init__(self, num_classes, device, opt):
        super().__init__('ResNext50', torchvision.models.resnext50_32x4d)
        self.model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        self.updateparams()
        self.model.to(device)
        self.defineOpt(opt)
        # self.defineTransform(232, 224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class ShuffleNetV205(GenericModel):
    def __init__(self, num_classes, device, opt):
        super().__init__('ShuffleNetV205', torchvision.models.shufflenet_v2_x0_5)
        self.model.fc = nn.Linear(in_features=1024, out_features=num_classes, bias=True)
        self.updateparams()
        self.model.to(device)
        self.defineOpt(opt)
        # self.defineTransform(256, 224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class SqueezeNet10(GenericModel):
    def __init__(self, num_classes, device, opt):
        super().__init__('SqueezeNet10', torchvision.models.squeezenet1_0)
        self.model.classifier[-3] = nn.Conv2d(512, num_classes, kernel_size=1)
        self.updateparams()
        self.model.to(device)
        self.defineOpt(opt)
        # self.defineTransform(256, 224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class VitB16(GenericModel):
    def __init__(self, num_classes, device, opt):
        super().__init__('VitB16', torchvision.models.vit_b_16)
        self.model.heads[-1] = nn.Linear(in_features=768, out_features=num_classes, bias=True)
        self.updateparams()
        self.model.to(device)
        self.defineOpt(opt)
        # self.defineTransform(256, 224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class Vgg11(GenericModel):
    def __init__(self, num_classes, device, opt):
        super().__init__('Vgg11', torchvision.models.vgg11)
        self.model.classifier[-1] = nn.Linear(in_features=4096, out_features=num_classes, bias=True)
        self.updateparams()
        self.model.to(device)
        self.defineOpt(opt)
        # self.defineTransform(256, 224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
