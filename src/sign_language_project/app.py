# src/sign_language_project/app.py
"""
Launches a Gradio interface to interact with the trained Sign Language MNIST model.
"""
import os
import argparse  # Import per gli argomenti da riga di comando
import torch
import gradio as gr
from PIL import Image

# Importiamo i moduli dal nostro progetto
from .model import SimpleCNN
from .dataset import get_transforms

# --- 1. CONFIGURAZIONE ---
MODEL_PATH = "results/models/sign_language_cnn.pth"
LABEL_MAP = {i: chr(65 + i) for i in range(25) if i != 9}
for i in range(9, 24):
    LABEL_MAP[i] = chr(65 + i + 1)


# --- 2. FUNZIONI HELPER ---
def load_model(model_path):
    """Carica il modello PyTorch addestrato e lo mette in modalità valutazione."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File del modello non trovato in {model_path}. Esegui prima il training.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=25).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict(input_dict):
    """
    Prende un DIZIONARIO da Gradio, lo processa e restituisce le predizioni.
    """
    if input_dict is None:
        return {"Nessun disegno": 1.0}

    image_array = input_dict.get("image", input_dict.get("composite"))

    if image_array is None:
        return {"Errore": "Non trovo l'immagine nell'input di Gradio."}

    inverted_image_array = 255 - image_array
    img_pil = Image.fromarray(inverted_image_array).convert("L")
    transform = get_transforms()
    tensor = transform(img_pil).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

    confidences = {LABEL_MAP[i]: float(prob) for i, prob in enumerate(probabilities)}
    return confidences


# --- 3. BLOCCO PRINCIPALE DI ESECUZIONE ---
if __name__ == '__main__':
    # Configura il parser per gli argomenti
    parser = argparse.ArgumentParser(description="Lancia l'interfaccia Gradio per il modello.")
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Esegui in modalità Docker (server name 0.0.0.0)."
    )
    args = parser.parse_args()

    # Determina il server_name in base all'argomento fornito
    # Se viene passato --docker, server_name sarà "0.0.0.0", altrimenti None.
    server_name_to_use = "0.0.0.0" if args.docker else None

    try:
        model = load_model(MODEL_PATH)

        iface = gr.Interface(
            fn=predict,
            inputs=gr.Paint(image_mode="L"),
            outputs=gr.Label(num_top_classes=3, label="Predizioni"),
            title="Classificatore Linguaggio dei Segni",
            description="Disegna una lettera (A-I, K-Y) nel riquadro e clicca 'Submit'."
        )

        print(f"Avvio del server Gradio... (Modalità Docker: {args.docker})")
        # Usa la variabile per lanciare il server
        iface.launch(server_name=server_name_to_use)

    except FileNotFoundError as e:
        print(e)
        gr.Interface(
            fn=lambda: str(e),
            inputs=[],
            outputs="text",
            title="ERRORE - MODELLO NON TROVATO",
        ).launch(server_name=server_name_to_use)

    except Exception as e:
        print(f"Un errore imprevisto ha bloccato l'avvio: {e}")
