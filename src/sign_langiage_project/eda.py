import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import os 

DATA_PATH = "data/"
RESULTS_PATH = "results/eda/"
TRAIN_FILE = os.path.join(DATA_PATH, "sign_mnist_train.csv")
TEST_FILE = OS.PATH.JOIN(DATA_PATH, "sign_mnist_test.csv")

def run_eda():
    """
    Esegue l'analisi esplorativa dei dati sul dataset Sign Language MNIST.
    Carica i dati, stampa informazioni di base e salva le visualizzazioni.
    """
    print("--- Inizio Analisi Esplorativa dei Dati (EDA) ---")

    os.makedirs(RESULTS_PATH, exist_ok=True)

    try:
        train_df = pd.read_csv(TRAIN_FILE)
        test_df = pd.read_csv(TEST_FILE)
        print(f"Dati di training caricati: {train_df.shape[0]} righe, {train_df.shape[1]} colonne.")
        print(f"Dati di test caricati: {test_df.shape[0]} righe, {test_df.shape[1]} colonne.")
    except FileNotFoundError:
        print(f"Errore: File non trovati in {DATA_PATH}.")
        print("Assicurati di aver scaricato 'sign_mnist_train.csv' e 'sign_mnist_test.csv' nella cartella 'data/'.")
        return

    plt.figure(figsize=(12, 6))
    sns.countplot(x='label', data=train_df, palette='viridis')
    plt.title('Distribuzione delle Classi nel Training Set')
    plt.xlabel('Etichetta (Lettera)')
    plt.ylabel('Conteggio')
    
    # Salva il grafico
    distribution_plot_path = os.path.join(RESULTS_PATH, "class_distribution.png")
    plt.savefig(distribution_plot_path)
    print(f"Grafico della distribuzione delle classi salvato in: {distribution_plot_path}")
    plt.close() 
    y_train = train_df['label']
    X_train = train_df.drop('label', axis=1)

    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    axes = axes.flatten()
    for i in range(10):
        idx = np.random.randint(0, len(X_train))
        image = X_train.iloc[idx].values.reshape(28, 28)
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f'Label: {y_train.iloc[idx]}')
        axes[i].axis('off')

    plt.tight_layout()
    sample_images_plot_path = os.path.join(RESULTS_PATH, "sample_images.png")
    plt.savefig(sample_images_plot_path)
    print(f"Grafico con immagini di esempio salvato in: {sample_images_plot_path}")
    plt.close()

    print("--- EDA Completata ---")

if __name__ == '__main__':
    # Questo blocco viene eseguito solo se lo script è lanciato direttamente
    run_eda()
    