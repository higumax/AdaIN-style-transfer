from torch.utils.data import Dataset
from PIL import Image
from glob import glob


class MyDataset(Dataset):
    def __init__(self, contents_path, styles_path, transform=None):
        super(Dataset, self).__init__()
        self.transform = transform
        ext = 'jpg'

        self.contents_path = glob(contents_path + '/*.' + ext)
        self.styles_path = glob(styles_path + '/*.' + ext)

        # Size of content images and style images needs to be same
        if len(self.contents_path) > len(self.styles_path):
            self.contents_path = self.contents_path[:len(self.styles_path)]
        else:
            self.styles_path = self.styles_path[:len(self.contents_path)]
        print(f"{len(self)} pairs of images are found.")

    def __len__(self):
        return len(self.contents_path)

    def __getitem__(self, idx):
        content_image_path, style_image_path = self.contents_path[idx], self.styles_path[idx]

        content_image = Image.open(content_image_path).convert('RGB')
        style_image = Image.open(style_image_path).convert('RGB')

        # preprocess image
        if self.transform:
            content_image = self.transform(content_image)
            style_image = self.transform(style_image)     

        return content_image, style_image