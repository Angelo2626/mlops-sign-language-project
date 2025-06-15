# MLOps Project: Sign Language MNIST Classifier

Questo progetto implementa una pipeline MLOps completa per l'addestramento e il deployment di un classificatore di immagini per il dataset Sign Language MNIST. L'obiettivo è dimostrare le best practice di sviluppo software e automazione nel contesto del Machine Learning.

Il progetto include un'interfaccia web interattiva costruita con Gradio per testare il modello disegnando direttamente nel browser.

## Struttura del Progetto

-   `.github/workflows/`: Contiene il workflow di GitHub Actions per la CI/CD.
-   `src/sign_language_project/`: Il codice sorgente del pacchetto Python.
    -   `train.py`: Script per l'addestramento del modello.
    -   `app.py`: Script per lanciare l'interfaccia web con Gradio.
    -   `model.py` e `dataset.py`: Moduli per il modello e la gestione dati.
-   `tests/`: Test unitari eseguiti con `pytest`.
-   `Dockerfile`: Definisce l'ambiente per il **training**.
-   `Dockerfile.app`: Definisce l'ambiente per l'**app Gradio**.
-   `requirements.txt`: Elenca le dipendenze Python.

## Funzionalità e Tecnologie Utilizzate

-   **Controllo di Versione:** Git e GitHub.
-   **Modello di Machine Learning:** Una CNN implementata con **PyTorch**.
-   **Containerizzazione:** **Docker** per creare ambienti isolati sia per il training che per l'inferenza.
-   **Continuous Integration (CI):** **GitHub Actions** per l'automazione di linting (`pylint`) e test unitari (`pytest`).
-   **Continuous Delivery (CD):** Il workflow CI/CD pubblica automaticamente le immagini Docker per il training e per l'app su **Docker Hub**.
-   **Interfaccia Utente:** Un'applicazione web interattiva creata con **Gradio** per permettere la predizione in tempo reale.

## Come Eseguire il Progetto

### Prerequisiti

-   Git
-   Docker Desktop

### Eseguire l'Applicazione Interattiva (Metodo Consigliato)

Il modo più semplice per testare il progetto è eseguire l'immagine Docker dell'applicazione, che è già stata pubblicata su Docker Hub e contiene il modello pre-addestrato.

1.  Esegui il seguente comando nel tuo terminale:
    ```bash
    # Sostituisci angelo2626 con il tuo username di Docker Hub
    docker run --rm -p 7860:7860 angelo2626/sign-language-app:latest
    ```
2.  Apri il browser e vai su `http://localhost:7860`.

### Eseguire il Training da Zero

1.  Clona la repository e naviga nella cartella.
2.  Scarica il [dataset da Kaggle](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) e posiziona i file CSV nella cartella `data/`.
3.  Esegui il training all'interno di un container Docker. Questo comando mappa la cartella dei dati per la lettura e la cartella dei risultati per salvare il modello addestrato.
    ```bash
    docker run --rm -v "${PWD}/data:/app/data" -v "${PWD}/results:/app/results" angelo2626/sign-language-trainer:latest
    ```

## Pipeline CI/CD

Il workflow in `.github/workflows/ci.yml` automatizza l'intero processo. Ad ogni `push` sul branch `main`:
1.  **Test Job:** Esegue `pylint` e `pytest` per validare il codice.
2.  **Build and Push Job:** Se i test passano, costruisce e pubblica le immagini `sign-language-trainer` e `sign-language-app` su Docker Hub.

---
