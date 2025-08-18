import torch

class VisionEncoderWrapper(torch.nn.Module):
    def __init__(self, vision_encoder):
        super().__init__()
        self.vision_encoder = vision_encoder
    def forward(self, x):
        return self.vision_encoder(x)