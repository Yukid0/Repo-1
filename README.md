# Classification d'actions à partir de squelettes NTU RGB+D

Ce projet implémente un pipeline complet pour la classification d'actions à partir du dataset NTU RGB+D. Il transforme les fichiers de squelettes 3D en images 2D, puis utilise un modèle de deep learning pour les classifier.

## Structure du projet

Le projet est organisé en plusieurs modules:

- `config.py`: Configuration globale (chemins, paramètres)
- `skeleton_parser.py`: Fonctions pour parser les fichiers .skeleton
- `preprocess.py`: Prétraitement des données (conversion squelettes → images)
- `model.py`: Définition du modèle de classification
- `train.py`: Entraînement du modèle
- `evaluate.py`: Évaluation du modèle
- `main.py`: Point d'entrée principal

## Prérequis

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib
- scikit-learn
- tqdm
- seaborn

Installation des dépendances:

```bash
pip install tensorflow opencv-python numpy matplotlib scikit-learn tqdm seaborn
```

## Configuration

Le projet utilise maintenant des chemins relatifs basés sur le répertoire personnel de l'utilisateur. Vous pouvez toujours personnaliser les chemins dans `config.py` :

```python
# Chemins des données
SKELETON_DATASET_PATH = os.path.join(BASE_PATH, "Datasets", "nturgbd_skeletons_s001_to_s017", "nturgb+d_skeletons")
PROCESSED_IMAGE_PATH = os.path.join(BASE_PATH, "Datasets", "ntu_images_processed")
MODEL_SAVE_PATH = os.path.join(BASE_PATH, "Models")
```

## Utilisation

### Exécution complète du pipeline

Pour exécuter toutes les étapes (prétraitement, entraînement, évaluation):

```bash
python main.py --all
```

### Exécution par étapes

Pour exécuter uniquement le prétraitement:

```bash
python main.py --preprocess
```

Pour exécuter uniquement l'entraînement:

```bash
python main.py --train
```

Pour exécuter uniquement l'évaluation:

```bash
python main.py --evaluate
```

### Choix du backbone

Vous pouvez spécifier le backbone à utiliser pour l'entraînement:

```bash
python main.py --train --backbone ResNet50
```

Options disponibles: `MobileNetV2` (par défaut), `ResNet50`, `EfficientNetB0`

## Détails du pipeline

### 1. Prétraitement

- Parse les fichiers .skeleton du dataset NTU RGB+D
- Convertit chaque séquence de squelettes en une image 2D
- Organise les images par classe dans des sous-dossiers
- Visualise des échantillons et affiche des statistiques sur le dataset
- Journalisation détaillée des opérations et des erreurs

### 2. Entraînement

- Charge les images prétraitées
- Applique de l'augmentation de données
- Entraîne un modèle CNN basé sur un backbone pré-entraîné
- Effectue un fine-tuning automatique des dernières couches
- Sauvegarde le meilleur modèle
- Affiche les courbes d'apprentissage
- Journalisation détaillée du processus d'entraînement

### 3. Évaluation

- Charge le modèle entraîné
- Évalue ses performances sur le dataset
- Affiche la matrice de confusion et des métriques détaillées
- Visualise des exemples de prédictions
- Sauvegarde automatiquement les résultats d'évaluation (graphiques et rapports)
- Journalisation détaillée des performances

## Gestion des erreurs

Le projet inclut maintenant une gestion robuste des erreurs:

- Vérification de l'existence des chemins et fichiers
- Journalisation détaillée via le module `logging`
- Gestion des exceptions pour chaque étape du pipeline
- Messages d'erreur clairs et informatifs

## Personnalisation

Vous pouvez modifier plusieurs paramètres dans `config.py`:

- `IMAGE_SIZE`: Taille des images générées
- `BATCH_SIZE`: Taille des batchs pour l'entraînement
- `EPOCHS`: Nombre d'époques d'entraînement
- `LEARNING_RATE`: Taux d'apprentissage
- `BACKBONE`: Architecture du backbone ("MobileNetV2", "ResNet50", "EfficientNetB0")
- `OVERWRITE_EXISTING`: Si `True`, réécrit les images déjà traitées

## Dataset NTU RGB+D

Le dataset NTU RGB+D est un large dataset pour la reconnaissance d'actions à partir de squelettes 3D. Il contient 60 classes d'actions différentes, réalisées par 40 sujets distincts.

Pour plus d'informations sur le dataset: [NTU RGB+D](https://rose1.ntu.edu.sg/dataset/actionRecognition/)

## Références

- Shahroudy, A., Liu, J., Ng, T. T., & Wang, G. (2016). NTU RGB+D: A large scale dataset for 3D human activity analysis. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1010-1019). 