from efficientnet_pytorch import EfficientNet
from main.custom_model.Darknet53 import darknet53
from main.custom_model.ResNet import ResNet, BasicBlock, BottleNeck

def EfficientNet_b7(num_classes=18):
    model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=num_classes)
    return model


def DarkNet53(num_classes=18):
    return darknet53(num_classes)


def ResNet18(num_classes=18):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def ResNet34(num_classes=18):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def ResNet50(num_classes=18):
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes)


def ResNet101(num_classes=18):
    return ResNet(BottleNeck, [3, 4, 23, 3], num_classes)


def ResNet152(num_classes=18):
    return ResNet(BottleNeck, [3, 8, 36, 3], num_classes)
