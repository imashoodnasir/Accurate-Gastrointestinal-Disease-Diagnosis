from torch.utils.data import Dataset
import os, cv2
import numpy as np

class SegDataset(Dataset):
    def __init__(self, root, split="train", img_size=352, transforms=None):
        self.root = root
        self.split = split
        self.img_dir = os.path.join(root, "images", split)
        self.mask_dir = os.path.join(root, "masks", split)
        self.ids = sorted([f for f in os.listdir(self.img_dir) if f.lower().endswith((".jpg", ".png"))])
        self.transforms = transforms

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        img = cv2.imread(os.path.join(self.img_dir, name))[:, :, ::-1]
        mask = cv2.imread(os.path.join(self.mask_dir, name), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)
        if self.transforms:
            out = self.transforms(image=img, mask=mask)
            img, mask = out["image"], out["mask"].unsqueeze(0).float()
        return {"image": img, "mask": mask, "id": name}

class ClsDataset(Dataset):
    def __init__(self, root, split="train", img_size=256, transforms=None, classes=None):
        self.root = root
        self.split = split
        self.img_dir = os.path.join(root, "images", split)
        self.classes = sorted(classes or os.listdir(self.img_dir))
        self.paths, self.labels = [], []
        for c, cls in enumerate(self.classes):
            d = os.path.join(self.img_dir, cls)
            for f in os.listdir(d):
                if f.lower().endswith((".jpg", ".png")):
                    self.paths.append(os.path.join(d,f))
                    self.labels.append(c)
        self.transforms = transforms

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = cv2.imread(path)[:, :, ::-1]
        if self.transforms:
            img = self.transforms(image=img)["image"]
        return {"image": img, "label": self.labels[idx], "id": os.path.basename(path)}
