from torch.utils.data import Dataset
import os
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, transform, mode="train"):
        self.transform = transform
        self.mode = mode

        if self.mode == "train":
            self.image_folder = os.listdir('datasets/train')
        elif self.mode == "test":
            self.image_folder = os.listdir('datasets/test')
        elif self.mode == "valid":
            self.image_folder = os.listdir('datasets/valid')

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, index):
        img_path = self.image_folder[index]
        img = Image.open(os.path.join('datasets', self.mode, img_path)).convert('RGB')
        img = self.transform(img)

        label = img_path.split('_')[-1].split('.')[0]
        label = int(label)
        return(img, label)