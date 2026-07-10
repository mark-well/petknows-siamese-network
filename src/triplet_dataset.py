from torch.utils.data import Dataset
from PIL import Image
import random
import os

class TripletDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.classes = os.listdir(root)
        self.images = {}

        for cls in self.classes:
            folder = os.path.join(root, cls)
            self.images[cls] = [
                os.path.join(folder, img)
                for img in os.listdir(folder)
            ]

    def __len__(self):
        return sum(len(images) for images in self.images.values())
    
    def __getitem__(self, index):
        # choose anchor class
        anchor_class = random.choice(self.classes)

        # choose another class
        negative_class = random.choice(
            [c for c in self.classes if c != anchor_class]
        )

        anchor_path, positive_path = random.sample(
            self.images[anchor_class], 2
        )

        negative_path = random.choice(
            self.images[negative_class]
        )

        anchor = Image.open(anchor_path).convert("RGB")
        positive = Image.open(positive_path).convert("RGB")
        negative = Image.open(negative_path).convert("RGB")

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative