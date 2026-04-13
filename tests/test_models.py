import torch
from src.models.architectures import ResNet50

def test_forward_pass():
    model = ResNet50(num_classes=10)
    x = torch.randn(2, 3, 224, 224)
    out = model(x.view(2, -1))
    assert out.shape == (2, 10)
