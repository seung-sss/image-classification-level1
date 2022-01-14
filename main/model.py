from efficientnet_pytorch import EfficientNet

def EfficientNet_b7(num_classes=18):
    model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=num_classes)
    return model