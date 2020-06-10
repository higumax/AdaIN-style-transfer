import torch 
import torch.nn as nn
from torchvision.models import vgg19
from pprint import pprint

class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.eps = 1e-5

    def forward(self, x, y):
        mean_x, mean_y = torch.mean(x, dim=(2, 3), keepdim=True), torch.mean(y, dim=(2, 3), keepdim=True)
        std_x, std_y = torch.std(x, dim=(2, 3), keepdim=True) + self.eps, torch.std(y, dim=(2, 3), keepdim=True) + self.eps
        return std_y * (x - mean_x) / std_x + mean_y

class Encoder(nn.Module):
    FEATURE_LAYERS = [1, 6, 11, 20] # relu1_1, relu2_1, relu3_1, relu4_1

    def __init__(self):
        super(Encoder, self).__init__()
        self.vgg19 = vgg19(pretrained=True)

        # Parameters of encoder are fixed
        for p in self.vgg19.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        features = []
        for idx, layer in self.vgg19.features._modules.items():
            idx = int(idx)
            x = layer(x)
            if idx in Encoder.FEATURE_LAYERS:
                features.append(x)
                if idx == Encoder.FEATURE_LAYERS[-1]:
                    break
        return features

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2.0, mode="nearest"),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2.0, mode="nearest"),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2.0, mode="nearest"),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )

    def forward(self, x):
        return self.model(x)
