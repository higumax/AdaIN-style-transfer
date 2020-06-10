import torch 
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from model import Encoder, Decoder, AdaptiveInstanceNorm2d
from dataset import MyDataset
import numpy as np
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

max_epoch = 100
batch_size = 20
alpha = 1.0

# Load dataset
transforms = torchvision.transforms.Compose([
    # smaller dimention of the image is matched to 512px while keeping the aspect ratio.
    torchvision.transforms.Resize(512),
    torchvision.transforms.RandomCrop((256, 256)),
    torchvision.transforms.ToTensor()
])
train_data = MyDataset("./data/content/", "./data/style", transforms)
train_data_loader = DataLoader(train_data, batch_size, shuffle=True)

# Load test image
test_content_img = Image.open("./imgs/bear.jpg")
test_content_img = transforms(test_content_img)
test_content_img = torch.unsqueeze(test_content_img, 0).to(device)
test_style_img = Image.open("./imgs/starry_night.jpg")
test_style_img = transforms(test_style_img)
test_style_img = torch.unsqueeze(test_style_img, 0).to(device)

# model, optimizer, loss function
enc = Encoder().to(device)
dec = Decoder().to(device)
adain = AdaptiveInstanceNorm2d().to(device)
loss_mse = torch.nn.MSELoss().to(device)
optimizer = optim.Adam(dec.parameters(), lr=1e-4)

for epoch in range(max_epoch):
    for i, (content, style) in enumerate(train_data_loader, 0):
        optimizer.zero_grad()

        content = content.to(device)
        style = style.to(device)

        content_f = enc(content)
        style_f = enc(style)

        normalized = adain(content_f[-1], style_f[-1])
        normalized = (1.0 - alpha) * content_f[-1] + alpha * normalized

        output = dec(normalized)
        output_f = enc(output)

        loss_content = loss_mse(normalized, output_f[-1])
        loss_style = 0.0
        for x, y in zip(style_f, output_f):
            loss_style += loss_mse(torch.mean(x, dim=(2, 3)), torch.mean(y, dim=(2, 3)))
            loss_style += loss_mse(torch.std(x, dim=(2, 3)), torch.std(y, dim=(2, 3)))
        
        total_loss = loss_content + 10 * loss_style
        total_loss.backward()

        optimizer.step()
    
        print(f"[epoch {epoch+1:2d}/{max_epoch}, data {i:3d}/{len(train_data_loader)}] loss: {total_loss.item():.3f}")
    
    torch.save(dec.state_dict(), f"weights/epoch{epoch+1:02d}.w")

    # test output
    content_f = enc(test_content_img)
    style_f = enc(test_style_img)
    normalized = adain(content_f[-1], style_f[-1])
    normalized = (1.0 - alpha) * content_f[-1] + alpha * normalized
    output = dec(normalized)
    output = output.clone().detach().to("cpu").numpy()
    output = (output[0].transpose(1, 2, 0)*255).clip(0, 255).astype(np.uint8)
    output_img = Image.fromarray(output)
    output_img.save(f'weights/epoch{epoch+1:02d}.jpg')
