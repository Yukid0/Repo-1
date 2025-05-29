import os
import argparse
import logging
from datetime import datetime
from tqdm import tqdm
import time

from config import SKELETON_DATASET_PATH, PROCESSED_IMAGE_PATH, BACKBONE
from preprocess import preprocess_skeletons, visualize_samples, analyze_dataset
from train import prepare_datasets, train_model
from evaluate import load_best_model, create_test_dataset, evaluate_model, plot_confusion_matrix, visualize_predictions

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    """
    Point d'entrée principal du programme.
    """
    # Analyser les arguments de la ligne de commande
    parser = argparse.ArgumentParser(description="Pipeline de traitement et classification de squelettes NTU RGB+D")
    
    parser.add_argument("--preprocess", action="store_true", help="Exécuter l'étape de prétraitement")
    parser.add_argument("--train", action="store_true", help="Exécuter l'étape d'entraînement")
    parser.add_argument("--evaluate", action="store_true", help="Exécuter l'étape d'évaluation")
    parser.add_argument("--all", action="store_true", help="Exécuter toutes les étapes")
    parser.add_argument("--backbone", type=str, choices=["MobileNetV2", "ResNet50", "EfficientNetB0"], 
                        default=BACKBONE, help="Architecture du backbone à utiliser")
    
    args = parser.parse_args()
    
    # Si aucune option n'est spécifiée, afficher l'aide
    if not (args.preprocess or args.train or args.evaluate or args.all):
        parser.print_help()
        return
    
    # Afficher une barre de progression globale
    total_steps = sum([args.preprocess or args.all, args.train or args.all, args.evaluate or args.all])
    
    with tqdm(total=total_steps, desc="Progression globale", unit="étape") as global_pbar:
        # Exécuter les étapes demandées
        if args.preprocess or args.all:
            run_preprocess()
            global_pbar.update(1)
        
        if args.train or args.all:
            run_train(backbone=args.backbone)
            global_pbar.update(1)
        
        if args.evaluate or args.all:
            run_evaluate()
            global_pbar.update(1)
            
    logger.info("Pipeline terminé avec succès!")

def run_preprocess():
    """
    Exécute l'étape de prétraitement.
    
    Returns:
        bool: True si le prétraitement a réussi, False sinon
    """
    logger.info("\n" + "="*80)
    logger.info("ÉTAPE 1: PRÉTRAITEMENT")
    logger.info("="*80)
    
    # Vérifier que le chemin du dataset existe
    if not os.path.exists(SKELETON_DATASET_PATH):
        logger.error(f"ERREUR: Le chemin du dataset {SKELETON_DATASET_PATH} n'existe pas!")
        return False
    
    logger.info(f"Début du prétraitement du dataset NTU RGB+D...")
    logger.info(f"- Source: {SKELETON_DATASET_PATH}")
    logger.info(f"- Destination: {PROCESSED_IMAGE_PATH}")
    
    # Lancer le prétraitement avec une barre de progression
    with tqdm(total=100, desc="Progression du prétraitement", unit="%") as pbar:
        # Lancer le prétraitement
        start_time = datetime.now()
        pbar.update(10)  # Mise à jour initiale
        
        sample_images, sample_labels = preprocess_skeletons(SKELETON_DATASET_PATH, PROCESSED_IMAGE_PATH)
        pbar.update(70)  # Mise à jour après le prétraitement principal
        
        end_time = datetime.now()
        
        logger.info(f"Temps de traitement total: {end_time - start_time}")
        
        # Vérifier si des échantillons ont été générés
        if not sample_images or not sample_labels:
            logger.warning("Aucun échantillon généré pendant le prétraitement.")
            return False
        
        # Visualiser les échantillons
        visualize_samples(sample_images, sample_labels)
        pbar.update(10)  # Mise à jour après la visualisation
        
        # Analyser le dataset
        class_counts = analyze_dataset(PROCESSED_IMAGE_PATH)
        pbar.update(10)  # Mise à jour finale
    
    return len(class_counts) > 0

def run_train(backbone=BACKBONE):
    """
    Exécute l'étape d'entraînement.
    
    Args:
        backbone (str): Architecture du backbone à utiliser
        
    Returns:
        bool: True si l'entraînement a réussi, False sinon
    """
    logger.info("\n" + "="*80)
    logger.info(f"ÉTAPE 2: ENTRAÎNEMENT (backbone: {backbone})")
    logger.info("="*80)
    
    # Vérifier que le répertoire des images traitées existe
    if not os.path.exists(PROCESSED_IMAGE_PATH):
        logger.error(f"ERREUR: Le répertoire des images traitées {PROCESSED_IMAGE_PATH} n'existe pas!")
        logger.error("Veuillez d'abord exécuter l'étape de prétraitement.")
        return False
    
    # Lancer l'entraînement avec une barre de progression
    with tqdm(total=100, desc="Progression de l'entraînement", unit="%") as pbar:
        logger.info("Préparation des datasets...")
        train_dataset, validation_dataset, num_classes = prepare_datasets()
        pbar.update(30)  # Mise à jour après la préparation des données
        
        if train_dataset and validation_dataset and num_classes > 0:
            logger.info(f"Entraînement du modèle avec backbone {backbone}...")
            model, history = train_model(train_dataset, validation_dataset, num_classes, backbone=backbone)
            pbar.update(60)  # Mise à jour après l'entraînement
            
            if model and history:
                logger.info("\nEntraînement terminé avec succès!")
                pbar.update(10)  # Mise à jour finale
                return True
        
        logger.error("Échec de l'entraînement.")
        return False

def run_evaluate():
    """
    Exécute l'étape d'évaluation.
    
    Returns:
        bool: True si l'évaluation a réussi, False sinon
    """
    logger.info("\n" + "="*80)
    logger.info("ÉTAPE 3: ÉVALUATION")
    logger.info("="*80)
    
    # Lancer l'évaluation avec une barre de progression
    with tqdm(total=100, desc="Progression de l'évaluation", unit="%") as pbar:
        # Charger le meilleur modèle
        model = load_best_model()
        pbar.update(20)  # Mise à jour après le chargement du modèle
        
        if not model:
            logger.error("Impossible de charger le modèle. Veuillez d'abord exécuter l'étape d'entraînement.")
            return False
        
        # Créer le dataset de test
        test_dataset, class_indices = create_test_dataset()
        pbar.update(20)  # Mise à jour après la création du dataset
        
        if not test_dataset:
            logger.error("Impossible de créer le dataset de test.")
            return False
        
        # Évaluer le modèle
        evaluation, y_true, y_pred = evaluate_model(model, test_dataset)
        pbar.update(30)  # Mise à jour après l'évaluation
        
        if not evaluation:
            logger.error("Échec de l'évaluation du modèle.")
            return False
        
        # Visualiser les résultats
        plot_confusion_matrix(y_true, y_pred, class_indices)
        pbar.update(15)  # Mise à jour après la matrice de confusion
        
        visualize_predictions(model, test_dataset)
        pbar.update(15)  # Mise à jour finale
        
        return True

if __name__ == "__main__":
    main() 