import torch
from torch.utils.data import DataLoader, Dataset
import os
import torch
from torchvision import transforms
from PIL import Image


'''The Data should be of this format--
|-- DATASET_NAME
|   |-- testing
|   |   |-- class 0
|   |   |-- class 1
|   |   |-- class 2
|   |   |-- class 3
|   |   |-- .
|   |   |-- .
|   |   |-- .
|   |   |-- .
|   |   |-- .
|   |   `-- class n
|   `-- training
|   |   |-- class 0
|   |   |-- class 1
|   |   |-- class 2
|   |   |-- class 3
|   |   |-- .
|   |   |-- .
|   |   |-- .
|   |   |-- .
|   |   |-- .
|   |   `-- class n'''
class ImageDataset(Dataset):
    def __init__(self, data_path, dataset_name, shape, split = "train"):
        self.data_path = data_path
        self.split = split
        if(dataset_name.lower() == "mnist"):
            self.mean = (0.1307,)
            self.std = (0.3081,)
        elif(dataset_name.lower()=="cifar10"):
            self.mean = (0.49139968, 0.48215827, 0.44653124)
            self.std = (0.24703233, 0.24348505, 0.26158768)
        self.class_id_map = {}
        if(split == "train"):
            self.base_path = os.path.join(data_path, "training")
            self.transform = transforms.Compose([
            transforms.Resize(shape),
            transforms.RandomRotation(10, fill=0),  
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=5,
                fill=0
            ),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
            ])
        else:
            self.base_path = os.path.join(data_path, "testing")
            self.transform = transforms.Compose([
            transforms.Resize(shape),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
            ])
        self.classes = os.listdir(self.base_path)
        self.idx_map = []

        for index, i in enumerate(self.classes):
            self.class_id_map[i] = index
            for j in os.listdir(os.path.join(self.base_path,i)):
                self.idx_map.append((i, j))
    def __len__(self):
        return len(self.idx_map)

    def __getitem__(self, idx):
        target_class, source_img_name = self.idx_map[idx]
        source_img_path = os.path.join(self.base_path, target_class, source_img_name)
        target_class_id = self.class_id_map[target_class]
        img = Image.open(source_img_path)
        img = self.transform(img)
        return {'input_img': torch.tensor(img, dtype = torch.float32),
                'target_class': torch.tensor(target_class_id, dtype = torch.long).unsqueeze(dim=0)}
    
def collate_fn(batch):
    inputs = torch.stack([b['input_img'] for b in batch], dim=0)
    targets = torch.stack([b['target_class'] for b in batch], dim=0)
    return inputs, targets



if __name__ == "__main__":
    batch_size = 32
    dataset = ImageDataset(data_path = f"{input('Enter data path: ')}", dataset_name = f"{input('Enter dataset name: ')}", shape = (32, 32))
    dataloader = DataLoader(dataset, batch_size = 32, shuffle = True, drop_last = True, collate_fn = collate_fn)
    for batch in dataloader:
        breakpoint()
    X = next(dataloader)