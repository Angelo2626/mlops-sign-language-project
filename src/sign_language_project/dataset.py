import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms 

class SignLanguageMNIST(Dataset):
    """Custom Dataset per il Sign Language MNIST da file CSV."""

    def __init__(self, csv_file, transform=None):
        """
        Args: 
            csv_file (string): Percorso al file CSV con annotazioni.
            transform (callable, optional): Trasformazione opzionale da applicare ad un campione.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform 

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = self.data_frame.iloc[idx, 0]
        image = self.data_frame.iloc[idx, 1:].values.astype('uint8').reshape((28, 28, 1))

        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)

def get_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])