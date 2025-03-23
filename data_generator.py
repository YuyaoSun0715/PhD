import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class CancerDataset(Dataset):
    def __init__(self, cancer_folders, img_size=(28, 28), num_samples_per_class=1, transform=None):
        self.cancer_folders = cancer_folders
        self.img_size = img_size
        self.num_samples_per_class = num_samples_per_class
        self.transform = transform
        self.image_paths = self._get_image_paths()

    def _get_image_paths(self):
        image_paths = []
        for folder in self.cancer_folders:
            images = [os.path.join(folder, img) for img in os.listdir(folder) if img.endswith('.png')]
            random.shuffle(images)
            image_paths.extend(images[:self.num_samples_per_class])
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')
        image = image.resize(self.img_size)
        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.tensor(image).unsqueeze(0)  # Add channel dimension
        return image

class SinusoidDataset(Dataset):
    def __init__(self, batch_size, num_samples_per_class, amp_range=(0.1, 5.0), phase_range=(0, np.pi), input_range=(-5.0, 5.0)):
        self.batch_size = batch_size
        self.num_samples_per_class = num_samples_per_class
        self.amp_range = amp_range
        self.phase_range = phase_range
        self.input_range = input_range
        self.data = self._generate_sinusoid_batch()

    def _generate_sinusoid_batch(self):
        amp = np.random.uniform(self.amp_range[0], self.amp_range[1], [self.batch_size])
        phase = np.random.uniform(self.phase_range[0], self.phase_range[1], [self.batch_size])
        inputs = np.random.uniform(self.input_range[0], self.input_range[1], [self.batch_size, self.num_samples_per_class, 1])
        outputs = amp[:, None] * np.sin(inputs - phase[:, None])
        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(outputs, dtype=torch.float32)

    def __len__(self):
        return self.batch_size

    def __getitem__(self, idx):
        return self.data[0][idx], self.data[1][idx]

class DataGenerator:
    def __init__(self, num_samples_per_class, batch_size, datasource='CancerCell', config={}):
        self.num_samples_per_class = num_samples_per_class
        self.batch_size = batch_size
        self.datasource = datasource
        self.img_size = config.get('img_size', (28, 28))
        self.num_classes = config.get('num_classes', 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if datasource == 'sinusoid':
            self.dataset = SinusoidDataset(batch_size, num_samples_per_class)
        elif datasource == 'CancerCell':
            data_folder = config.get('data_folder', './data')
            cancer_folders = [os.path.join(data_folder, family, cell)
                                 for family in os.listdir(data_folder)
                                 if os.path.isdir(os.path.join(data_folder, family))
                                 for cell in os.listdir(os.path.join(data_folder, family))]
            random.seed(1)
            random.shuffle(cancer_folders)
            num_train = config.get('num_train', 1200) - 100
            self.metatrain_cancer_folders = cancer_folders[:num_train]
            self.metaval_cancer_folders = cancer_folders[num_train:num_train+100]
            self.dataset = CancerDataset(self.metatrain_cancer_folders, self.img_size, num_samples_per_class)
        else:
            raise ValueError('Unrecognized data source')

    def get_dataloader(self, train=True):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
