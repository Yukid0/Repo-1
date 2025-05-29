import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import logging
import random

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Importer les modules du projet
from config import SKELETON_DATASET_PATH
from skeleton_parser import parse_skeleton_file, sequence_to_image, get_label_from_filename

# Dictionnaire des noms d'actions (120 classes)
ACTION_NAMES = {
    0: "Boire de l'eau",
    1: "Manger",
    2: "Se brosser les dents",
    3: "Se brosser les cheveux",
    4: "Laisser tomber",
    5: "Ramasser",
    6: "Lancer",
    7: "S'asseoir",
    8: "Se lever",
    9: "Applaudir",
    10: "Lire",
    11: "Écrire",
    12: "Déchirer du papier",
    13: "Porter des lunettes",
    14: "Enlever des lunettes",
    15: "Mettre un chapeau/casquette",
    16: "Enlever un chapeau/casquette",
    17: "Mettre une veste",
    18: "Enlever une veste",
    19: "Mettre une chaussure",
    20: "Enlever une chaussure",
    21: "Encourager (cheer up)",
    22: "Saluer de la main",
    23: "Donner un coup de pied",
    24: "Mettre quelque chose dans sa poche",
    25: "Sortir quelque chose de sa poche",
    26: "Sauter",
    27: "Faire un mouvement de téléphone",
    28: "Jouer sur son téléphone",
    29: "Taper sur un clavier",
    30: "Pointer vers quelque chose",
    31: "Prendre un selfie",
    32: "Vérifier l'heure (montre)",
    33: "Frotter les mains",
    34: "Hocher la tête",
    35: "Faire un signe de non avec la tête",
    36: "Faire un signe d'au revoir",
    37: "Faire un geste de poing",
    38: "Faire un geste 'OK'",
    39: "Faire un geste de pouce vers le haut",
    40: "Faire un geste de pouce vers le bas",
    41: "Faire un geste de 'V' (victoire)",
    42: "Faire un geste de 'cœur' avec les doigts",
    43: "Toucher autre chose (avec les doigts)",
    44: "Faire un signe de 'pistolet' avec la main",
    45: "Toucher la tête (mal de tête)",
    46: "Toucher la poitrine (douleur à la poitrine)",
    47: "Toucher le dos (mal de dos)",
    48: "Toucher le cou (mal de cou)",
    49: "Se frotter l'estomac (mal d'estomac)",
    50: "Arc du bras (étirement)",
    51: "Rotation des articulations (exercice)",
    52: "Faire des sauts sur place",
    53: "Sauter en écartant les jambes",
    54: "Faire des squats",
    55: "Faire des pompes",
    56: "Faire des abdominaux",
    57: "Faire du jogging",
    58: "Marcher",
    59: "Frapper (boxe)",
    60: "Taper dans les mains",
    61: "Faire un bras de fer (main droite)",
    62: "Faire un bras de fer (main gauche)",
    63: "Dire 'Arrête' avec la main",
    64: "Taper du pied",
    65: "Plier les bras",
    66: "Lever un haltère (avec un bras)",
    67: "Lever un haltère (avec deux bras)",
    68: "Faire une pompe à une main (droite)",
    69: "Faire une pompe à une main (gauche)",
    70: "Faire une pompe à deux mains",
    71: "Faire une pompe avec les poings serrés",
    72: "Parler au téléphone",
    73: "Jouer avec un téléphone/tablette",
    74: "Taper sur un clavier",
    75: "Jouer de la guitare",
    76: "Jouer du violon",
    77: "Jouer du violoncelle",
    78: "Toucher un autre personne (poignée de main, etc.)",
    79: "Donner un objet à une autre personne",
    80: "Toucher le sol",
    81: "Mettre une cravate",
    82: "Attacher les lacets",
    83: "Mettre une collection",
    84: "Mettre des chaussettes",
    85: "Mettre une chemise",
    86: "Mettre une veste",
    87: "Mettre un chapeau",
    88: "Mettre des lunettes",
    89: "Mettre des gants",
    90: "Mettre un écharpe",
    91: "Enlever une cravate",
    92: "Enlever les lacets",
    93: "Enlever une collection",
    94: "Enlever des chaussettes",
    95: "Enlever une chemise",
    96: "Enlever une veste",
    97: "Enlever un chapeau",
    98: "Enlever des lunettes",
    99: "Enlever des gants",
    100: "Enlever un écharpe",
    101: "Essuyer la table",
    102: "Plier du papier",
    103: "Plier des vêtements",
    104: "Verser de l'eau",
    105: "Verser du lait",
    106: "Verser du café",
    107: "Décorer un gâteau",
    108: "Faire un sandwich",
    109: "Faire un gâteau",
    110: "Mettre de la vaisselle dans l'armoire",
    111: "Ouvrir un conteneur",
    112: "Vérifier un conteneur",
    113: "Fermer un conteneur",
    114: "Vérifier les poches",
    115: "Allumer une bougie",
    116: "Souffler une bougie",
    117: "Mettre quelque chose dans un conteneur",
    118: "Sortir quelque chose d'un conteneur",
    119: "Travailler sur un ordinateur"
}

def get_action_name(label):
    """
    Obtient le nom de l'action à partir du numéro de label.
    
    Args:
        label (int): Numéro de label (base 0)
        
    Returns:
        str: Nom de l'action ou "Action inconnue" si le label n'est pas reconnu
    """
    return ACTION_NAMES.get(label, f"Action inconnue ({label})")

def test_skeleton_to_rgb(skeleton_file_path):
    """
    Teste la conversion d'un fichier squelette en image RGB
    et affiche le résultat avec une visualisation des canaux séparés.
    
    Args:
        skeleton_file_path (str): Chemin vers un fichier .skeleton
    """
    logger.info(f"Traitement du fichier: {os.path.basename(skeleton_file_path)}")
    
    # Extraire le label
    label = get_label_from_filename(os.path.basename(skeleton_file_path))
    action_name = get_action_name(label)
    logger.info(f"Label extrait: {label} - {action_name}")
    
    # Parser le fichier squelette
    skeleton_data = parse_skeleton_file(skeleton_file_path)
    if skeleton_data is None:
        logger.error("Échec du parsing du fichier squelette")
        return
    
    logger.info(f"Données de squelette chargées: {skeleton_data.shape}")
    
    # Convertir en image RGB
    rgb_image = sequence_to_image(skeleton_data)
    if rgb_image is None:
        logger.error("Échec de la conversion en image RGB")
        return
    
    logger.info(f"Image RGB générée: {rgb_image.shape}")
    
    # Visualiser l'image et ses canaux
    plt.figure(figsize=(15, 10))
    
    # Image RGB complète
    plt.subplot(2, 2, 1)
    plt.imshow(rgb_image)
    plt.title("Image RGB complète")
    plt.axis('off')
    
    # Canal R (coordonnées X)
    plt.subplot(2, 2, 2)
    plt.imshow(rgb_image[:, :, 0], cmap='Reds')
    plt.title("Canal R (coordonnées X)")
    plt.axis('off')
    
    # Canal G (coordonnées Y)
    plt.subplot(2, 2, 3)
    plt.imshow(rgb_image[:, :, 1], cmap='Greens')
    plt.title("Canal G (coordonnées Y)")
    plt.axis('off')
    
    # Canal B (coordonnées Z)
    plt.subplot(2, 2, 4)
    plt.imshow(rgb_image[:, :, 2], cmap='Blues')
    plt.title("Canal B (coordonnées Z)")
    plt.axis('off')
    
    plt.suptitle(f"Visualisation de l'image RGB générée\n{action_name} (A{label+1:03d})")
    plt.tight_layout()
    plt.show()
    
    return rgb_image

def find_sample_skeleton_files(input_dir, num_samples=3):
    """
    Trouve quelques fichiers .skeleton d'exemple dans le répertoire d'entrée.
    
    Args:
        input_dir (str): Répertoire contenant les fichiers .skeleton
        num_samples (int): Nombre d'échantillons à trouver
        
    Returns:
        list: Liste des chemins vers les fichiers .skeleton d'exemple
    """
    try:
        all_files = [f for f in os.listdir(input_dir) if f.endswith('.skeleton')]
        if not all_files:
            logger.warning(f"Aucun fichier .skeleton trouvé dans {input_dir}")
            return []
        
        # Prendre des échantillons aléatoires
        sample_files = random.sample(all_files, min(num_samples, len(all_files)))
        return [os.path.join(input_dir, f) for f in sample_files]
        
    except Exception as e:
        logger.error(f"Erreur lors de la recherche de fichiers d'exemple: {e}")
        return []

if __name__ == "__main__":
    # Vérifier que le chemin du dataset existe
    if not os.path.exists(SKELETON_DATASET_PATH):
        logger.error(f"ERREUR: Le chemin du dataset {SKELETON_DATASET_PATH} n'existe pas!")
        logger.info(f"Veuillez configurer le chemin correct dans config.py")
    else:
        logger.info(f"Chemin du dataset trouvé: {SKELETON_DATASET_PATH}")
        
        # Lister quelques fichiers dans le répertoire pour vérifier
        files = os.listdir(SKELETON_DATASET_PATH)[:5]
        logger.info(f"Quelques fichiers dans le répertoire: {files}")
        
        # Trouver quelques fichiers d'exemple
        sample_files = find_sample_skeleton_files(SKELETON_DATASET_PATH)
        
        if not sample_files:
            logger.error("Aucun fichier d'exemple trouvé pour le test")
        else:
            logger.info(f"Fichiers d'exemple trouvés: {len(sample_files)}")
            
            # Tester chaque fichier
            for file_path in sample_files:
                logger.info(f"\n{'='*50}")
                logger.info(f"Test du fichier: {os.path.basename(file_path)}")
                logger.info(f"{'='*50}")
                
                # Convertir en image RGB et visualiser
                test_skeleton_to_rgb(file_path) 