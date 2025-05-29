import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import logging
from tqdm import tqdm

from config import (
    PROCESSED_IMAGE_PATH, 
    IMAGE_SIZE, 
    BATCH_SIZE, 
    EPOCHS, 
    MODEL_SAVE_PATH,
    BACKBONE
)
from model import create_model, unfreeze_layers

logger = logging.getLogger(__name__)

# Classe TqdmCallback pour afficher une barre de progression pendant l'entraînement
class TqdmProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, epochs, verbose=1):
        super(TqdmProgressCallback, self).__init__()
        self.epochs = epochs
        self.verbose = verbose
        
    def on_train_begin(self, logs=None):
        self.pbar = tqdm(total=self.epochs, desc="Entraînement", 
                         unit="epoch", position=0, leave=True,
                         bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Mettre à jour la barre de progression avec les métriques
        metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
        self.pbar.set_postfix_str(metrics_str)
        self.pbar.update(1)
        
    def on_train_end(self, logs=None):
        self.pbar.close()

def prepare_datasets():
    """
    Prépare les datasets d'entraînement et de validation.
    
    Returns:
        train_dataset: Dataset d'entraînement
        validation_dataset: Dataset de validation
        num_classes: Nombre de classes détectées
    """
    # Vérifier que le répertoire des images existe
    if not os.path.exists(PROCESSED_IMAGE_PATH):
        logger.error(f"Le répertoire {PROCESSED_IMAGE_PATH} n'existe pas.")
        return None, None, 0
        
    # Augmentation des données pour l'ensemble d'entraînement
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )

    # Pas d'augmentation pour la validation, juste rescaling
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    # Création des datasets
    try:
        logger.info("Chargement des données d'entraînement...")
        with tqdm(total=100, desc="Chargement des données d'entraînement", unit="%") as pbar:
            train_dataset = train_datagen.flow_from_directory(
                PROCESSED_IMAGE_PATH,
                target_size=IMAGE_SIZE,
                batch_size=BATCH_SIZE,
                color_mode='rgb',
                class_mode='sparse',
                subset='training',
                seed=42
            )
            pbar.update(50)  # Mise à jour de la barre de progression à 50%
            
            validation_dataset = val_datagen.flow_from_directory(
                PROCESSED_IMAGE_PATH,
                target_size=IMAGE_SIZE,
                batch_size=BATCH_SIZE,
                color_mode='rgb',
                class_mode='sparse',
                subset='validation',
                seed=42
            )
            pbar.update(50)  # Compléter la barre de progression
        
        num_classes = len(train_dataset.class_indices)
        logger.info(f"Nombre de classes détectées: {num_classes}")
        logger.info(f"Nombre d'échantillons d'entraînement: {train_dataset.samples}")
        logger.info(f"Nombre d'échantillons de validation: {validation_dataset.samples}")
        
        return train_dataset, validation_dataset, num_classes
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données: {e}")
        return None, None, 0

def setup_callbacks(model_name):
    """
    Configure les callbacks pour l'entraînement.
    
    Args:
        model_name: Nom de base pour le modèle sauvegardé
        
    Returns:
        callbacks: Liste de callbacks
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_path = os.path.join(MODEL_SAVE_PATH, f"{model_name}_{timestamp}.h5")
    log_dir = os.path.join(MODEL_SAVE_PATH, f"logs/{model_name}_{timestamp}")
    
    # Créer le répertoire des logs si nécessaire
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)
    
    # Définir les callbacks
    model_checkpoint = ModelCheckpoint(
        filepath=model_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    tensorboard = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1
    )
    
    # Ajouter notre callback de barre de progression
    tqdm_callback = TqdmProgressCallback(epochs=EPOCHS)
    
    return [early_stopping, reduce_lr, model_checkpoint, tensorboard, tqdm_callback]

def plot_training_history(history):
    """
    Trace les courbes d'apprentissage (précision et perte).
    
    Args:
        history: Historique d'entraînement du modèle
    """
    try:
        # Précision
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Précision du modèle')
        plt.ylabel('Précision')
        plt.xlabel('Epoch')
        plt.legend(['Entraînement', 'Validation'], loc='lower right')
        
        # Perte
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Perte du modèle')
        plt.ylabel('Perte')
        plt.xlabel('Epoch')
        plt.legend(['Entraînement', 'Validation'], loc='upper right')
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logger.error(f"Erreur lors de l'affichage des courbes d'apprentissage: {e}")

def train_model(train_dataset, validation_dataset, num_classes, backbone=BACKBONE, do_fine_tuning=True):
    """
    Entraîne le modèle sur les données.
    
    Args:
        train_dataset: Dataset d'entraînement
        validation_dataset: Dataset de validation
        num_classes: Nombre de classes
        backbone: Architecture du backbone à utiliser
        do_fine_tuning: Si True, effectue un fine-tuning après l'entraînement initial
        
    Returns:
        model: Modèle entraîné
        history: Historique d'entraînement
    """
    if num_classes == 0:
        logger.error("Impossible de créer le modèle: aucune classe détectée")
        return None, None
    
    # Créer le modèle
    try:
        logger.info(f"Création du modèle avec backbone {backbone}...")
        with tqdm(total=100, desc="Création du modèle", unit="%") as pbar:
            model = create_model(backbone=backbone, num_classes=num_classes)
            pbar.update(100)
        model.summary()
    except Exception as e:
        logger.error(f"Erreur lors de la création du modèle: {e}")
        return None, None
    
    # Configurer les callbacks
    callbacks = setup_callbacks(f"skeleton_{backbone}")
    
    # Entraînement initial avec backbone gelé
    logger.info("\nDébut de l'entraînement initial...")
    try:
        history = model.fit(
            train_dataset,
            epochs=EPOCHS,
            validation_data=validation_dataset,
            callbacks=callbacks
        )
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement initial: {e}")
        return None, None
    
    # Fine-tuning (optionnel)
    if do_fine_tuning:
        logger.info("\nDébut du fine-tuning...")
        try:
            with tqdm(total=100, desc="Préparation du fine-tuning", unit="%") as pbar:
                model = unfreeze_layers(model)
                pbar.update(100)
                
            fine_tune_callbacks = setup_callbacks(f"skeleton_{backbone}_finetuned")
            
            # Entraînement avec backbone partiellement dégelé
            fine_tune_history = model.fit(
                train_dataset,
                epochs=10,
                validation_data=validation_dataset,
                callbacks=fine_tune_callbacks
            )
            
            # Fusionner les historiques
            for key in history.history:
                history.history[key].extend(fine_tune_history.history[key])
        except Exception as e:
            logger.warning(f"Erreur lors du fine-tuning: {e}")
            logger.info("Utilisation du modèle sans fine-tuning.")
    
    return model, history

if __name__ == "__main__":
    logger.info("Préparation des datasets...")
    train_dataset, validation_dataset, num_classes = prepare_datasets()
    
    if train_dataset and validation_dataset and num_classes > 0:
        logger.info(f"Entraînement du modèle avec backbone {BACKBONE}...")
        model, history = train_model(train_dataset, validation_dataset, num_classes)
        
        if model and history:
            logger.info("\nEntraînement terminé!")
            plot_training_history(history)
    else:
        logger.error("Impossible de procéder à l'entraînement: problème avec les datasets") 