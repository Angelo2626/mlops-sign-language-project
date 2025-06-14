"""Main script for training the SimpleCNN model."""
import os  # 1. Standard library

import torch  # 2. Third-party libraries
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import SignLanguageMNIST, get_transforms  # 3. Local application
from .model import SimpleCNN

# Definiamo i percorsi e i parametri
DATA_PATH = "data/"
TRAIN_FILE = os.path.join(DATA_PATH, "sign_mnist_train.csv")
MODEL_SAVE_PATH = "results/models/"
MODEL_NAME = "sign_language_cnn.pth"

# Parametri di training
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 50

def train_model():
    """Funzione principale per addestrare il modello."""
    print("--- Inizio Training del Modello ---")

    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training su dispositivo: {device}")

    # 1. Caricamento dei dati
    transforms = get_transforms()
    train_dataset = SignLanguageMNIST(csv_file=TRAIN_FILE, transform=transforms)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Inizializzazione del modello, loss function e optimizer
    model = SimpleCNN(num_classes=25).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Ciclo di training
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        loop = tqdm(train_loader, leave=True)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(loss=loss.item())

        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}")

    # 4. Salvataggio del modello
    full_model_path = os.path.join(MODEL_SAVE_PATH, MODEL_NAME)
    torch.save(model.state_dict(), full_model_path)
    print(f"Modello addestrato salvato in: {full_model_path}")

    print("--- Training Completato ---")

if __name__ == '__main__':
    train_model()
