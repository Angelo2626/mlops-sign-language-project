# MLOps Project: Sign Language MNIST Classifier

Questo progetto implementa una pipeline MLOps completa per l'addestramento e il deployment di un classificatore di immagini per il dataset Sign Language MNIST. L'obiettivo è dimostrare le best practice di sviluppo software e automazione nel contesto del Machine Learning.

## Struttura del Progetto

Il progetto è organizzato seguendo una struttura standard per pacchetti Python:

-   `.github/workflows/`: Contiene i workflow di GitHub Actions per la CI/CD.
-   `data/`: (Locale, ignorata da git) Contiene i file CSV del dataset.
-   `src/sign_language_project/`: Il codice sorgente del pacchetto Python.
    -   `dataset.py`: Gestione del caricamento dati con `torch.utils.data.Dataset`.
    -   `model.py`: Definizione dell'architettura della rete neurale convoluzionale (CNN).
    -   `train.py`: Script per l'orchestrazione del training.
    -   `eda.py`: Script per l'analisi esplorativa dei dati.
-   `tests/`: Contiene i test unitari per il progetto, eseguiti con `pytest`.
-   `Dockerfile`: Definisce l'ambiente containerizzato per il training.
-   `requirements.txt`: Elenca le dipendenze Python del progetto.
-   `pytest.ini`: File di configurazione per `pytest`.

## Funzionalità e Tecnologie Utilizzate

-   **Controllo di Versione:** Git e GitHub per la gestione del codice e della collaborazione.
-   **Modello di Machine Learning:** Una Rete Neurale Convoluzionale (CNN) implementata con **PyTorch** per classificare le 25 lettere del linguaggio dei segni.
-   **Containerizzazione:** **Docker** per creare un ambiente di training isolato e riproducibile.
-   **Continuous Integration (CI):** **GitHub Actions** per l'automazione di:
    -   **Linting** del codice con `pylint` per garantire la qualità e lo stile.
    -   **Test Unitari** con `pytest` per verificare la correttezza del modello.
-   **Continuous Delivery (CD):** Il workflow CI/CD pubblica automaticamente l'immagine Docker su **Docker Hub** dopo che i test sono passati con successo.

## Come Eseguire il Progetto

### Prerequisiti

-   Git
-   Python 3.11
-   Conda (consigliato per la gestione dell'ambiente)
-   Docker Desktop

### Esecuzione Locale

1.  Clonare la repository:
    ```bash
    git clone https://github.com/Angelo2626/mlops-sign-language-project.git
    cd mlops-sign-language-project
    ```
2.  Creare e attivare l'ambiente virtuale:
    ```bash
    conda create --name mlops_project python=3.11
    conda activate mlops_project
    ```
3.  Installare le dipendenze:
    ```bash
    pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
    ```
4.  Scaricare il [dataset da Kaggle](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) e posizionare i file `sign_mnist_train.csv` e `sign_mnist_test.csv` nella cartella `data/`.

5.  Eseguire il training:
    ```bash
    python -m src.sign_language_project.train
    ```

### Esecuzione con Docker

1.  Costruire l'immagine Docker:
    ```bash
    docker build -t sign-language-trainer .
    ```
2.  Eseguire il training all'interno del container. Il modello addestrato (`.pth`) verrà salvato nella cartella locale `results/models`.
    ```bash
    docker run --rm -v "${PWD}/results:/app/results" sign-language-trainer
    ```

## Pipeline CI/CD

Il workflow definito in `.github/workflows/ci.yml` automatizza l'intero processo. Ad ogni `push` sul branch `main`, vengono eseguiti i seguenti job:
1.  **Lint & Test:** Verifica la qualità del codice.
2.  **Build and Push:** Se i test passano, l'immagine Docker viene costruita e pubblicata su Docker Hub.

L'immagine pubblicata è disponibile qui: `https://hub.docker.com/r/angelo2626/mlops-sign-language-project`

---
