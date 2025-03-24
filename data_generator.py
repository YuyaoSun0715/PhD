import os
import random
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset

class MetaDataset(Dataset):
    def __init__(self, root_dir, ways=5, shots=1, queries=15, transform=None):
        """
        Args:
            root_dir: path to the dataset folder with one folder per class.
            ways: number of classes per task.
            shots: number of support examples per class.
            queries: number of query examples per class.
            transform: torchvision transforms to apply (default converts image to tensor).
        """
        self.root_dir = root_dir
        self.ways = ways
        self.shots = shots
        self.queries = queries
        self.transform = transform if transform is not None else transforms.ToTensor()

        # List class folders (each subfolder is a class)
        self.class_folders = [os.path.join(root_dir, cls) for cls in os.listdir(root_dir)
                              if os.path.isdir(os.path.join(root_dir, cls))]
        self.class_to_images = {}
        for folder in self.class_folders:
            cls_name = os.path.basename(folder)
            images = [os.path.join(folder, img) for img in os.listdir(folder)
                      if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.class_to_images[cls_name] = images
        self.classes = list(self.class_to_images.keys())

    def __len__(self):
        # Arbitrarily large since tasks are sampled on the fly.
        return 1000000

    def __getitem__(self, idx):
        # Randomly sample "ways" classes for the task.
        selected_classes = random.sample(self.classes, self.ways)
        support_images, support_labels = [], []
        query_images, query_labels = [], []
        label_map = {cls: i for i, cls in enumerate(selected_classes)}
        for cls in selected_classes:
            imgs = self.class_to_images[cls]
            # Ensure there are enough images per class.
            assert len(imgs) >= self.shots + self.queries, f"Not enough images in class {cls}"
            selected_imgs = random.sample(imgs, self.shots + self.queries)
            for i in range(self.shots):
                img = Image.open(selected_imgs[i]).convert('RGB')
                support_images.append(self.transform(img))
                support_labels.append(label_map[cls])
            for i in range(self.shots, self.shots + self.queries):
                img = Image.open(selected_imgs[i]).convert('RGB')
                query_images.append(self.transform(img))
                query_labels.append(label_map[cls])
        support_images = torch.stack(support_images)  # shape: [ways*shots, C, H, W]
        query_images = torch.stack(query_images)      # shape: [ways*queries, C, H, W]
        support_labels = torch.tensor(support_labels)
        query_labels = torch.tensor(query_labels)
        return support_images, support_labels, query_images, query_labels
