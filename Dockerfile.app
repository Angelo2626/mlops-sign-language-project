# Dockerfile per l'app Gradio
FROM python:3.11-slim

WORKDIR /app

# Copia le dipendenze e installale
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copia il codice sorgente
COPY ./src ./src

# Copia il modello addestrato che abbiamo generato in precedenza
# Questo presuppone che il file sia presente nella cartella results/models/
# durante la fase di build dell'immagine.
COPY ./results/models/sign_language_cnn.pth ./results/models/sign_language_cnn.pth

# Esponi la porta che Gradio usa di default
EXPOSE 7860

# Comando per lanciare l'app in modalità Docker
CMD ["python", "-m", "src.sign_language_project.app", "--docker"]
