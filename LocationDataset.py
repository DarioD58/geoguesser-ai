from typing import Any, Callable, Optional, Union
from torch import Tensor
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import os


class LocationDataset(Dataset):

    def __init__(self, labels_file: str, img_dir: str, transform: Optional[Callable]=None, 
                target_transform: Optional[Callable]=None, orientation: str = None) -> None:
        self.targets = pd.read_csv(labels_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.orientation = orientation

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> Union[tuple[Any, float, float], tuple[list[Any], float, float]]:
        if self.orientation:
            img_path = os.path.join(self.img_dir, self.targets.iloc[index, 0], f"{self.orientation}.jpg")
            image = Image.open(img_path)

            if self.transform:
                image = self.transform(image)

        else:
            image = []
            img_path = os.path.join(self.img_dir, self.targets.iloc[index, 0])
            for filename in os.listdir(img_path):
                img= Image.open(os.path.join(img_path, filename))

                if self.transform:
                    img = self.transform(img)

                image.append(img)
        
        latitude = self.targets.iloc[index, 1]
        longitude = self.targets.iloc[index, 2]

        if self.target_transform:
            latitude = self.target_transform(latitude)
            longitude = self.target_transform(longitude)
            

        return image, latitude, longitude