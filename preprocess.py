import os
import cv2
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import logging

from config import (
    SKELETON_DATASET_PATH, 
    PROCESSED_IMAGE_PATH, 
    OVERWRITE_EXISTING,
    NUM_SAMPLES_TO_VISUALIZE
)
from skeleton_parser import parse_skeleton_file, sequence_to_image, get_label_from_filename

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def preprocess_skeletons(input_dir, output_dir, overwrite=OVERWRITE_EXISTING):
    """
    Prétraite tous les fichiers .skeleton dans le répertoire d'entrée
    et sauvegarde les images résultantes dans le répertoire de sortie.
    """
    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    # Lister tous les fichiers .skeleton
    try:
        all_files = [f for f in os.listdir(input_dir) if f.endswith('.skeleton')]
        logger.info(f"Nombre total de fichiers .skeleton trouvés: {len(all_files)}")
        if not all_files:
            logger.warning(f"Aucun fichier .skeleton trouvé dans {input_dir}")
            return [], []
    except Exception as e:
        logger.error(f"Erreur lors de la lecture du répertoire {input_dir}: {e}")
        return [], []
    
    # Variables pour visualisation
    sample_images = []
    sample_labels = []
    
    # Traiter chaque fichier
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    # Créer une barre de progression avec plus d'informations
    progress_bar = tqdm(all_files, desc="Conversion des squelettes en images", 
                        unit="fichier", ncols=100, 
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    
    for filename in progress_bar:
        try:
            # Mettre à jour la description avec le nom du fichier actuel
            progress_bar.set_description(f"Traitement: {filename}")
            
            # 1. Extraire le label
            label = get_label_from_filename(filename)
            if label == -1:
                logger.warning(f"Impossible d'extraire le label de {filename}, fichier ignoré")
                error_count += 1
                continue
            
            # 2. Créer le sous-dossier pour cette classe si besoin
            class_folder = os.path.join(output_dir, str(label))
            os.makedirs(class_folder, exist_ok=True)
            
            # 3. Définir le chemin de l'image de sortie
            output_image_path = os.path.join(class_folder, filename.replace('.skeleton', '.png'))
            
            # 4. Vérifier si l'image existe déjà
            if not overwrite and os.path.exists(output_image_path):
                skipped_count += 1
                # Mettre à jour la barre de progression
                progress_bar.set_postfix(traités=processed_count, ignorés=skipped_count, erreurs=error_count)
                continue
                
            # 5. Appliquer le pipeline de transformation
            skeleton_filepath = os.path.join(input_dir, filename)
            skeleton_data = parse_skeleton_file(skeleton_filepath)
            
            if skeleton_data is None:
                error_count += 1
                # Mettre à jour la barre de progression
                progress_bar.set_postfix(traités=processed_count, ignorés=skipped_count, erreurs=error_count)
                continue
                
            image_data = sequence_to_image(skeleton_data)
            
            # 6. Sauvegarder l'image sur le disque
            if image_data is not None:
                # Pour les images RGB, OpenCV utilise BGR par défaut, donc convertir
                cv2.imwrite(output_image_path, cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR))
                processed_count += 1
                
                # Collecter quelques échantillons pour visualisation
                if len(sample_images) < NUM_SAMPLES_TO_VISUALIZE:
                    sample_images.append(image_data)
                    sample_labels.append(label)
                    
                # Mettre à jour la barre de progression
                progress_bar.set_postfix(traités=processed_count, ignorés=skipped_count, erreurs=error_count)
        
        except Exception as e:
            logger.error(f"Erreur lors du traitement de {filename}: {e}")
            error_count += 1
            # Mettre à jour la barre de progression
            progress_bar.set_postfix(traités=processed_count, ignorés=skipped_count, erreurs=error_count)
    
    logger.info(f"\nTraitement terminé:")
    logger.info(f"- {processed_count} fichiers traités avec succès")
    logger.info(f"- {skipped_count} fichiers ignorés (déjà traités)")
    logger.info(f"- {error_count} fichiers avec erreurs")
    logger.info(f"Les images ont été sauvegardées dans: {output_dir}")
    
    return sample_images, sample_labels

def visualize_samples(sample_images, sample_labels):
    """
    Visualise des échantillons d'images générées à partir des squelettes.
    """
    if not sample_images or not sample_labels:
        logger.info("Aucun échantillon disponible pour visualisation.")
        return
        
    try:
        plt.figure(figsize=(15, 3))
        for i, (img, label) in enumerate(zip(sample_images, sample_labels)):
            plt.subplot(1, len(sample_images), i+1)
            plt.imshow(img)  # Afficher en RGB
            plt.title(f"Classe {label}")
            plt.axis('off')
        plt.suptitle("Exemples d'images RGB générées à partir des squelettes")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logger.error(f"Erreur lors de la visualisation des échantillons: {e}")

def analyze_dataset(processed_dir):
    """
    Analyse le dataset traité et affiche des statistiques et visualisations.
    """
    if not os.path.exists(processed_dir):
        logger.error(f"Le répertoire {processed_dir} n'existe pas.")
        return {}
        
    classes = [c for c in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, c))]
    
    if not classes:
        logger.warning("Aucune classe trouvée dans le répertoire de sortie.")
        return {}
        
    try:
        class_counts = {}
        
        # Ajouter une barre de progression pour l'analyse des classes
        for class_name in tqdm(classes, desc="Analyse des classes", unit="classe"):
            class_dir = os.path.join(processed_dir, class_name)
            class_counts[class_name] = len([f for f in os.listdir(class_dir) if f.endswith('.png')])
        
        # Afficher les statistiques
        logger.info(f"Nombre total de classes: {len(class_counts)}")
        logger.info(f"Nombre total d'images: {sum(class_counts.values())}")
        logger.info(f"Moyenne d'images par classe: {sum(class_counts.values()) / len(class_counts):.1f}")
        logger.info(f"Min: {min(class_counts.values())} images, Max: {max(class_counts.values())} images")
        
        # Visualiser la distribution des classes (top 20)
        top_classes = sorted(class_counts.items(), key=lambda x: int(x[0]))[:20]
        plt.figure(figsize=(12, 6))
        plt.bar([f"A{int(c[0])+1:03d}" for c in top_classes], [c[1] for c in top_classes])
        plt.title("Distribution des 20 premières classes")
        plt.xlabel("Classe")
        plt.ylabel("Nombre d'échantillons")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        return class_counts
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse du dataset: {e}")
        return {}

if __name__ == "__main__":
    # Vérifier que le chemin du dataset existe
    if not os.path.exists(SKELETON_DATASET_PATH):
        logger.error(f"ERREUR: Le chemin du dataset {SKELETON_DATASET_PATH} n'existe pas!")
    else:
        logger.info(f"Début du prétraitement du dataset NTU RGB+D...")
        logger.info(f"- Source: {SKELETON_DATASET_PATH}")
        logger.info(f"- Destination: {PROCESSED_IMAGE_PATH}")
        
        # Lancer le prétraitement
        start_time = datetime.now()
        sample_images, sample_labels = preprocess_skeletons(SKELETON_DATASET_PATH, PROCESSED_IMAGE_PATH)
        end_time = datetime.now()
        
        logger.info(f"Temps de traitement total: {end_time - start_time}")
        
        # Visualiser les échantillons
        visualize_samples(sample_images, sample_labels)
        
        # Analyser le dataset
        analyze_dataset(PROCESSED_IMAGE_PATH) 