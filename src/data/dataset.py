import os
from PIL import Image
from torch.utils.data import Dataset

class BrainTumor(Dataset):
    """
    Dataset for Kaggle Brain Tumor MRI.
    Will look for either 'train'/'test' or 'Training'/'Testing' folders.
    Directory structure under root_dir should be:
        root_dir/
          ├── train/ or Training/
          │     ├── glioma/
          │     ├── meningioma/
          │     ├── pituitary/
          │     └── no_tumor/
          └── test/ or Testing/
                ├── glioma/
                ├── meningioma/
                ├── pituitary/
                └── no_tumor/
    """
    CLASS_NAMES = ['glioma', 'meningioma', 'pituitary', 'notumor']
    FALLBACK = {'train': 'Training', 'test': 'Testing'}

    def __init__(self, root_dir, split='train', transform=None):
        self.transform = transform
        self.images = []
        self.labels = []

        # first try lowercase split folder
        split_dir = os.path.join(root_dir, split)
        # if that doesn't exist, fall back to uppercase naming
        if not os.path.isdir(split_dir):
            fallback = self.FALLBACK.get(split)
            if fallback:
                split_dir = os.path.join(root_dir, fallback)

        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Could not find folder for split '{split}': {split_dir}")

        # gather images
        for idx, cls in enumerate(self.CLASS_NAMES):
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                raise FileNotFoundError(f"Expected class directory not found: {cls_dir}")
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.images.append(os.path.join(cls_dir, fname))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('L')  # grayscale

        if self.transform:
            image = self.transform(image)

        return image, label
