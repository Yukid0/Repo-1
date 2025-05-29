import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import logging
from datetime import datetime
from tqdm import tqdm

from config import PROCESSED_IMAGE_PATH, IMAGE_SIZE, BATCH_SIZE, MODEL_SAVE_PATH

logger = logging.getLogger(__name__)

def load_best_model(model_dir=MODEL_SAVE_PATH):
    """
    Charge le meilleur modèle sauvegardé (basé sur la date de modification).
    
    Args:
        model_dir: Répertoire contenant les modèles sauvegardés
        
    Returns:
        model: Modèle Keras chargé ou None en cas d'erreur
    """
    if not os.path.exists(model_dir):
        logger.error(f"Le répertoire des modèles {model_dir} n'existe pas.")
        return None
        
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.h5')]
    
    if not model_files:
        logger.error(f"Aucun modèle trouvé dans {model_dir}")
        return None
    
    # Trier par date de modification (le plus récent en premier)
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
    
    # Charger le modèle le plus récent
    model_path = os.path.join(model_dir, model_files[0])
    logger.info(f"Chargement du modèle: {model_path}")
    
    try:
        with tqdm(total=100, desc="Chargement du modèle", unit="%") as pbar:
            model = load_model(model_path)
            pbar.update(100)
        logger.info("Modèle chargé avec succès")
        return model
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {e}")
        return None

def create_test_dataset():
    """
    Crée un dataset de test à partir des images traitées.
    
    Returns:
        test_dataset: Dataset de test
        class_indices: Dictionnaire des indices de classe
    """
    if not os.path.exists(PROCESSED_IMAGE_PATH):
        logger.error(f"Le répertoire des images {PROCESSED_IMAGE_PATH} n'existe pas.")
        return None, None
        
    # Générateur d'images pour le test (sans augmentation)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    try:
        logger.info("Chargement des données de test...")
        with tqdm(total=100, desc="Chargement des données de test", unit="%") as pbar:
            test_dataset = test_datagen.flow_from_directory(
                PROCESSED_IMAGE_PATH,
                target_size=IMAGE_SIZE,
                batch_size=BATCH_SIZE,
                color_mode='rgb',
                class_mode='sparse',
                shuffle=False  # Important pour la matrice de confusion
            )
            pbar.update(100)
        
        logger.info(f"Dataset de test créé avec {test_dataset.samples} échantillons")
        return test_dataset, test_dataset.class_indices
    
    except Exception as e:
        logger.error(f"Erreur lors de la création du dataset de test: {e}")
        return None, None

def evaluate_model(model, test_dataset):
    """
    Évalue le modèle sur le dataset de test.
    
    Args:
        model: Modèle Keras à évaluer
        test_dataset: Dataset de test
        
    Returns:
        evaluation: Résultats d'évaluation
        y_true: Étiquettes réelles
        y_pred: Prédictions
    """
    if model is None or test_dataset is None:
        logger.error("Modèle ou dataset de test invalide")
        return None, None, None
    
    try:
        # Évaluer le modèle
        logger.info("Évaluation du modèle...")
        with tqdm(total=100, desc="Évaluation du modèle", unit="%") as pbar:
            evaluation = model.evaluate(test_dataset)
            pbar.update(50)
            
            # Prédire sur le dataset de test
            logger.info("Génération des prédictions...")
            y_pred_proba = model.predict(test_dataset, verbose=0)
            pbar.update(50)
            
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Obtenir les vraies étiquettes
        y_true = test_dataset.classes
        
        logger.info(f"\nPrécision sur le dataset de test: {evaluation[1]:.4f}")
        
        return evaluation, y_true, y_pred
    
    except Exception as e:
        logger.error(f"Erreur lors de l'évaluation du modèle: {e}")
        return None, None, None

def plot_confusion_matrix(y_true, y_pred, class_indices, save_path=None):
    """
    Trace la matrice de confusion.
    
    Args:
        y_true: Étiquettes réelles
        y_pred: Prédictions
        class_indices: Dictionnaire des indices de classe
        save_path: Chemin pour sauvegarder la figure (optionnel)
    """
    if y_true is None or y_pred is None:
        logger.error("Données d'évaluation invalides")
        return
        
    try:
        # Créer la matrice de confusion
        logger.info("Création de la matrice de confusion...")
        with tqdm(total=100, desc="Création de la matrice de confusion", unit="%") as pbar:
            cm = confusion_matrix(y_true, y_pred)
            pbar.update(50)
            
            # Inverser le dictionnaire class_indices pour obtenir les noms de classe
            class_names = {v: f"A{k+1:03d}" for k, v in class_indices.items()}
            pbar.update(50)
        
        # Tracer la matrice de confusion
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=[class_names.get(i, i) for i in range(len(class_names))],
            yticklabels=[class_names.get(i, i) for i in range(len(class_names))]
        )
        plt.xlabel('Prédiction')
        plt.ylabel('Réalité')
        plt.title('Matrice de confusion')
        plt.tight_layout()
        
        # Sauvegarder la figure si un chemin est fourni
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Matrice de confusion sauvegardée dans {save_path}")
            
        plt.show()
        
        # Afficher le rapport de classification
        logger.info("Génération du rapport de classification...")
        report = classification_report(
            y_true, 
            y_pred, 
            target_names=[class_names.get(i, f"Class {i}") for i in range(len(class_names))],
            output_dict=True
        )
        
        # Sauvegarder le rapport si un chemin est fourni
        if save_path:
            report_path = os.path.splitext(save_path)[0] + "_report.txt"
            with open(report_path, 'w') as f:
                f.write("Rapport de classification:\n\n")
                f.write(classification_report(
                    y_true, 
                    y_pred, 
                    target_names=[class_names.get(i, f"Class {i}") for i in range(len(class_names))]
                ))
            logger.info(f"Rapport de classification sauvegardé dans {report_path}")
            
        logger.info("\nRapport de classification:")
        logger.info(classification_report(
            y_true, 
            y_pred, 
            target_names=[class_names.get(i, f"Class {i}") for i in range(len(class_names))]
        ))
        
        return report
        
    except Exception as e:
        logger.error(f"Erreur lors de la création de la matrice de confusion: {e}")

def visualize_predictions(model, test_dataset, num_samples=5, save_path=None):
    """
    Visualise quelques prédictions du modèle.
    
    Args:
        model: Modèle Keras
        test_dataset: Dataset de test
        num_samples: Nombre d'échantillons à visualiser
        save_path: Chemin pour sauvegarder la figure (optionnel)
    """
    if model is None or test_dataset is None:
        logger.error("Modèle ou dataset de test invalide")
        return
        
    try:
        # Réinitialiser le dataset
        test_dataset.reset()
        
        # Obtenir quelques échantillons
        logger.info(f"Visualisation de {num_samples} prédictions...")
        with tqdm(total=100, desc="Préparation des visualisations", unit="%") as pbar:
            batch_x, batch_y = next(test_dataset)
            batch_x = batch_x[:num_samples]
            batch_y = batch_y[:num_samples]
            pbar.update(50)
            
            # Faire des prédictions
            predictions = model.predict(batch_x, verbose=0)
            pred_classes = np.argmax(predictions, axis=1)
            pbar.update(50)
        
        # Inverser le dictionnaire class_indices pour obtenir les noms de classe
        class_names = {v: f"A{int(k)+1:03d}" for k, v in test_dataset.class_indices.items()}
        
        # Afficher les images avec les prédictions
        plt.figure(figsize=(15, 3))
        for i in range(num_samples):
            plt.subplot(1, num_samples, i+1)
            plt.imshow(batch_x[i])
            
            # Colorier en vert si correct, en rouge si incorrect
            color = 'green' if pred_classes[i] == batch_y[i] else 'red'
            
            plt.title(f"Réel: {class_names.get(batch_y[i], batch_y[i])}\nPréd: {class_names.get(pred_classes[i], pred_classes[i])}", 
                    color=color)
            plt.axis('off')
        
        plt.suptitle("Prédictions du modèle")
        plt.tight_layout()
        
        # Sauvegarder la figure si un chemin est fourni
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Visualisation des prédictions sauvegardée dans {save_path}")
            
        plt.show()
        
    except Exception as e:
        logger.error(f"Erreur lors de la visualisation des prédictions: {e}")

if __name__ == "__main__":
    # Configurer le logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Créer un dossier pour les résultats
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = os.path.join(MODEL_SAVE_PATH, f"evaluation_results_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Charger le meilleur modèle
    model = load_best_model()
    
    if model:
        # Créer le dataset de test
        test_dataset, class_indices = create_test_dataset()
        
        if test_dataset:
            # Évaluer le modèle
            evaluation, y_true, y_pred = evaluate_model(model, test_dataset)
            
            if evaluation is not None:
                # Visualiser les résultats
                confusion_matrix_path = os.path.join(results_dir, "confusion_matrix.png")
                plot_confusion_matrix(y_true, y_pred, class_indices, save_path=confusion_matrix_path)
                
                predictions_path = os.path.join(results_dir, "predictions.png")
                visualize_predictions(model, test_dataset)
                
                logger.info(f"Évaluation terminée. Résultats sauvegardés dans {results_dir}")
            else:
                logger.error("Échec de l'évaluation du modèle")
        else:
            logger.error("Impossible de créer le dataset de test")
    else:
        logger.error("Impossible de charger le modèle") 