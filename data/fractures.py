import os
from PIL import Image
import torch as t
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.utils as vutils


class FractureImages(Dataset):
    def __init__(self, root_dir, sub_dir, transform=None):
        self.root_dir = root_dir
        self.sub_dir = sub_dir
        self.transform = transform
        self.image_files = []


        # Collect all image file paths from the subdirectory
        sub_dir_path = os.path.join(self.root_dir, self.sub_dir)
        for file_name in os.listdir(sub_dir_path):
            if os.path.isfile(os.path.join(sub_dir_path, file_name)):
                self.image_files.append(os.path.join(sub_dir_path, file_name))

        # Load all images into memory
        self.samples = []
        for img_path in self.image_files:
            image = Image.open(img_path).convert('L')  # Convert to grayscale if needed
            if self.transform:
                image = self.transform(image)

            if image.shape[0] != 1:
                image = image.unsqueeze(0)
            self.samples.append(image)
        
        # Stack all samples into a single tensor
        self.samples = t.stack(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], 0  # placeholder label
    
    def save_images_as_grid(self, save_path):
        # Create a grid of images
        grid = vutils.make_grid(self.samples, nrow=8, padding=0, normalize=True)
        grid_image = transforms.ToPILImage()(grid)
        grid_image.save(save_path)

