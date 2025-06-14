import os
import torch
import gradio as gr
from PIL import Image
import numpy as np

from .model import SimpleCNN
from .dataset import get_transforms

MODEL_PATH = "results/models/sign_language_cnn.pth"
# Mapping delle etichette numeriche alle lettere (J=9, Z=25 non sono nel dataset)
LABEL_MAP = {i: chr(65 + i) for i in range(25) if i != 9}
# Aggiustiamo le etichette dopo la J
for i in range(9, 24):
    LABEL_MAP[i] = chr(65 + i + 1)

def load_model(model_path):
    """Carica il modello PyTorch addestrato."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File del modello non trovato in {model_path}. Esegui prima il training.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=25).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict(image):
    """
    Esegue la predizione su un'immagine di input (dal canvas di Gradio).
    Restituisce un dizionario di probabilità per ogni classe.
    """
    # L'input di Gradio Sketchpad è un array NumPy (H, W, C)
    # Convertiamolo in un'immagine PIL in scala di grigi
    img_pil = Image.fromarray(image).convert("L")

    # Applichiamo le stesse trasformazioni del training
    transform = get_transforms()
    tensor = transform(img_pil).unsqueeze(0) # Aggiunge la dimensione del batch

    with torch.no_grad():
        outputs = model(tensor)
        # Applichiamo Softmax per ottenere le probabilità
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

    # Creiamo il dizionario di output per Gradio
    confidences = {LABEL_MAP[i]: float(prob) for i, prob in enumerate(probabilities)}
    return confidences

# --- Caricamento e Lancio dell'Interfaccia ---
try:
    model = load_model(MODEL_PATH)

    # Creiamo l'interfaccia Gradio
    iface = gr.Interface(
        fn=predict,
        inputs=gr.Sketchpad(image_mode="RGB", shape=(28, 28), invert_colors=True, source="canvas"),
        outputs=gr.Label(num_top_classes=3, label="Predizioni"),
        title="Classificatore Linguaggio dei Segni (MNIST)",
        description="Disegna una lettera (A-I, K-Y) nel riquadro. Il modello proverà a riconoscerla."
    )

    # Lancia l'interfaccia
    iface.launch(server_name="0.0.0.0")

except FileNotFoundError as e:
    print(e)
    # Lancia un'interfaccia di errore se il modello non è presente
    gr.Interface(lambda: None, inputs=None, outputs="text", title="ERRORE", description=str(e)).launch()
