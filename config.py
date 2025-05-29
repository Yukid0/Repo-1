import os
import platform
from pathlib import Path

# Détection du répertoire de l'utilisateur
HOME = str(Path.home())

# Chemins des données - adaptés pour être plus portables
if platform.system() == "Windows":
    BASE_PATH = os.path.join(HOME, "Desktop", "Cours")
else:
    BASE_PATH = os.path.join(HOME, "Documents", "Cours")

# Chemins des données
SKELETON_DATASET_PATH = r"C:\Users\gross\Desktop\Cours\Datasets\nturgbd_skeletons_s001_to_s017\nturgb+d_skeletons"
PROCESSED_IMAGE_PATH = os.path.join(BASE_PATH, "Datasets", "ntu_images_processed")
MODEL_SAVE_PATH = os.path.join(BASE_PATH, "Models")

# Paramètres de prétraitement
IMAGE_SIZE = (128, 128)
OVERWRITE_EXISTING = False
NUM_SAMPLES_TO_VISUALIZE = 5

# Paramètres d'entraînement
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
BACKBONE = "MobileNetV2"  # Options: "MobileNetV2", "ResNet50", "EfficientNetB0"

# Créer les dossiers nécessaires
os.makedirs(PROCESSED_IMAGE_PATH, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True) 