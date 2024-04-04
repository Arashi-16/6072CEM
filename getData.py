from torch.utils.data import Dataset
from PIL import Image
import os

class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None, prefix='E:/archive/crop'):
        self.dataframe = dataframe
        self.transform = transform
        self.prefix = prefix

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = os.path.join(self.prefix, self.dataframe.iloc[idx, 0])  # Add prefix path
        label = self.dataframe.iloc[idx, 1]

        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label
