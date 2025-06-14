# src/sign_language_project/dataset.py
"""Handles the creation of the custom PyTorch Dataset for Sign Language MNIST."""
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image  # Import necessario

class SignLanguageMNIST(Dataset):
    """Custom Dataset per il Sign Language MNIST da file CSV."""

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Percorso al file CSV con annotazioni.
            transform (callable, optional): Trasformazione opzionale da applicare a un campione.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = self.data_frame.iloc[idx, 0]
        # Estraiamo i dati come array numpy
        image_array = self.data_frame.iloc[idx, 1:].values.astype('uint8')

        # Convertiamo l'array 1D in un'immagine PIL 2D in scala di grigi (L)
        img_pil = Image.fromarray(image_array.reshape(28, 28), mode='L')

        # Applichiamo le trasformazioni all'immagine PIL
        if self.transform:
            image_tensor = self.transform(img_pil)
        else:
            # Se non ci sono trasformazioni, convertiamo manualmente in tensore
            image_tensor = transforms.ToTensor()(img_pil)

        return image_tensor, torch.tensor(label, dtype=torch.long)


def get_transforms():
    """Returns the image transformation pipeline."""
    return transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
