import sys
import torch 
import torchvision
from model import Encoder, Decoder, AdaptiveInstanceNorm2d
import numpy as np
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

alpha = 1.0

transforms = torchvision.transforms.Compose([
    # smaller dimention of the image is matched to 512px while keeping the aspect ratio.
    torchvision.transforms.Resize((256, 256)),
    #torchvision.transforms.RandomCrop((256, 256)),
    torchvision.transforms.ToTensor()
])

# Load test image
test_content_img = Image.open(sys.argv[2])
test_content_img = transforms(test_content_img)
test_content_img = torch.unsqueeze(test_content_img, 0).to(device)
test_style_img = Image.open(sys.argv[3])
test_style_img = transforms(test_style_img)
test_style_img = torch.unsqueeze(test_style_img, 0).to(device)

# model, optimizer, loss function
enc = Encoder().to(device)
dec = Decoder().to(device)
adain = AdaptiveInstanceNorm2d().to(device)

dec.load_state_dict(torch.load(sys.argv[1]))

with torch.no_grad():
    content = test_content_img.to(device)
    style = test_style_img.to(device)

    content_f = enc(content)
    style_f = enc(style)

    normalized = adain(content_f[-1], style_f[-1])
    normalized = (1.0 - alpha) * content_f[-1] + alpha * normalized

    output = dec(normalized)

output = output.clone().detach().to("cpu").numpy()
output = (output[0].transpose(1, 2, 0)*255).clip(0, 255).astype(np.uint8)
output_img = Image.fromarray(output)
output_img.save('test.jpg')