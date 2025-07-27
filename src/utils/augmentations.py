from torchvision import transforms as t


grey_mean = [0.5]
grey_std = [0.5]

def get_grey_transforms():

  return t.Compose([t.Resize((224, 224)),
                    t.RandomHorizontalFlip(),
                    t.RandomVerticalFlip(),
                    t.RandomRotation(15),
                    t.ColorJitter(brightness=0.2, contrast=0.1),
                    t.ToTensor(),
                    t.Normalize(mean=grey_mean, std=grey_std),
                    ])    


def get_val_transforms():
    return t.Compose([
        t.Resize((224, 224)),
        t.ToTensor(),
        t.Normalize(grey_mean, grey_std),
        ])