# pylint: disable=C0103
"""Performs Exploratory Data Analysis (EDA) on the dataset."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Definiamo i percorsi relativi alla root del progetto
DATA_PATH = "data/"
RESULTS_PATH = "results/eda/"
TRAIN_FILE = os.path.join(DATA_PATH, "sign_mnist_train.csv")
TEST_FILE = os.path.join(DATA_PATH, "sign_mnist_test.csv")

def run_eda():
    """
    Esegue l'analisi esplorativa dei dati sul dataset Sign Language MNIST.
    Carica i dati, stampa informazioni di base e salva le visualizzazioni.
    """
    print("--- Inizio Analisi Esplorativa dei Dati (EDA) ---")

    os.makedirs(RESULTS_PATH, exist_ok=True)

    # 1. Caricamento dei dati
    try:
        train_df = pd.read_csv(TRAIN_FILE)
        _ = pd.read_csv(TEST_FILE)  # Carichiamo test_df per completezza ma lo ignoriamo
        print(f"Dati di training caricati: {train_df.shape[0]} righe, {train_df.shape[1]} colonne.")
    except FileNotFoundError:
        print(f"Errore: File non trovati in {DATA_PATH}.")
        print("Assicurati di aver scaricato 'sign_mnist_train.csv' e 'sign_mnist_test.csv'.")
        return

    # 2. Analisi della distribuzione delle classi (label)
    plt.figure(figsize=(12, 6))
    sns.countplot(x='label', data=train_df, palette='viridis')
    plt.title('Distribuzione delle Classi nel Training Set')
    plt.xlabel('Etichetta (Lettera)')
    plt.ylabel('Conteggio')

    distribution_plot_path = os.path.join(RESULTS_PATH, "class_distribution.png")
    plt.savefig(distribution_plot_path)
    print(f"Grafico della distribuzione salvato in: {distribution_plot_path}")
    plt.close()

    # 3. Visualizzazione di alcuni campioni
    y_train = train_df['label']
    X_train = train_df.drop('label', axis=1)

    # Usiamo l'underscore per la variabile 'fig' non usata
    _, axes = plt.subplots(2, 5, figsize=(15, 7))
    axes = axes.flatten()
    for i in range(10):
        idx = np.random.randint(0, len(X_train))
        image = X_train.iloc[idx].values.reshape(28, 28)
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f'Label: {y_train.iloc[idx]}')
        axes[i].axis('off')

    plt.tight_layout()
    # Ho spezzato la riga lunga in due per risolvere C0301
    sample_images_plot_path = os.path.join(
        RESULTS_PATH, "sample_images.png"
    )
    plt.savefig(sample_images_plot_path)
    print(f"Grafico con esempi salvato in: {sample_images_plot_path}")
    plt.close()

    print("--- EDA Completata ---")

if __name__ == '__main__':
    run_eda()
