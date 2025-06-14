# Usa un'immagine Python ufficiale e leggera come base
FROM python:3.11-slim

# Imposta la directory di lavoro all'interno del container
WORKDIR /app

# Copia prima il file delle dipendenze per sfruttare la cache di Docker
COPY requirements.txt .

# Installa le dipendenze, usando PyPI come default e aggiungendo l'URL di PyTorch
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copia il codice sorgente del progetto
COPY ./src ./src

# Definisci il comando da eseguire all'avvio del container
CMD ["python", "-m", "src.sign_language_project.train"]