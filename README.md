# ActionClassifierNTURGB+D

Ce projet implémente un pipeline complet pour la classification d'actions à partir du dataset NTU RGB+D. Il transforme les fichiers de squelettes 3D en images RGB, puis utilise un modèle de deep learning pour les classifier.

## Structure du projet

Le projet est organisé en plusieurs modules:

- `config.py`: Configuration globale (chemins, paramètres)
- `skeleton_parser.py`: Fonctions pour parser les fichiers .skeleton
- `preprocess.py`: Prétraitement des données (conversion squelettes → images RGB)
- `model.py`: Définition du modèle de classification
- `train.py`: Entraînement du modèle
- `evaluate.py`: Évaluation du modèle
- `main.py`: Point d'entrée principal
- `test_visualization.py`: Script pour tester la visualisation des squelettes

## Prérequis

- Python 3.7+
- TensorFlow 2.10.0 (recommandé pour stabilité)
- OpenCV
- NumPy
- Matplotlib
- scikit-learn
- tqdm
- seaborn

Installation des dépendances:

```bash
pip install tensorflow==2.10.0 opencv-python numpy matplotlib scikit-learn tqdm seaborn
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
- Convertit chaque séquence de squelettes en une image RGB 2D:
  - Canal R: coordonnées X des articulations
  - Canal G: coordonnées Y des articulations
  - Canal B: coordonnées Z des articulations
- Transpose les données pour obtenir des lignes horizontales dans les images
- Organise les images par classe dans des sous-dossiers
- Visualise des échantillons et affiche des statistiques sur le dataset
- Journalisation détaillée des opérations et des erreurs
- Affiche une barre de progression pour suivre l'avancement

### 2. Entraînement

- Charge les images prétraitées
- Applique de l'augmentation de données
- Entraîne un modèle CNN basé sur un backbone pré-entraîné
- Effectue un fine-tuning automatique des dernières couches
- Sauvegarde le meilleur modèle
- Affiche les courbes d'apprentissage
- Journalisation détaillée du processus d'entraînement
- Affiche une barre de progression pour suivre l'avancement

### 3. Évaluation

- Charge le modèle entraîné
- Évalue ses performances sur le dataset
- Affiche la matrice de confusion et des métriques détaillées
- Visualise des exemples de prédictions
- Sauvegarde automatiquement les résultats d'évaluation (graphiques et rapports)
- Journalisation détaillée des performances
- Affiche une barre de progression pour suivre l'avancement

## Représentation des squelettes en images RGB

Notre approche transforme les squelettes 3D en images RGB où:
- Chaque ligne horizontale représente une frame de la séquence
- Chaque colonne représente une articulation spécifique
- Les canaux R, G, B encodent respectivement les coordonnées X, Y, Z

Cette représentation permet:
- De conserver toute l'information spatiale 3D
- D'exploiter efficacement les architectures CNN existantes
- De visualiser intuitivement les mouvements dans l'espace

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

Le dataset NTU RGB+D est un large dataset pour la reconnaissance d'actions à partir de squelettes 3D. Il contient:
- Version NTU RGB+D 120: 120 classes d'actions

Mon projet prend en charge les 120 classes avec un dictionnaire complet des noms d'actions.


## Résultats

Les résultats préliminaires montrent:
- Précision globale: ~85% sur les 120 premières classes
- Temps d'inférence: ~20ms par séquence
- Performances variables selon la complexité des actions

Les actions avec des mouvements distincts (comme "s'asseoir" ou "se lever") sont généralement mieux classifiées que les actions subtiles (comme "écrire" ou "jouer avec son téléphone").

## Contributions et améliorations

Principales contributions de ce projet:
- Représentation RGB efficace des squelettes 3D
- Support complet des 120 classes du dataset NTU RGB+D
- Pipeline de bout en bout avec barres de progression
- Visualisation intuitive des squelettes pour le débogage

Améliorations futures possibles:
- Intégration d'architectures basées sur les transformers
- Exploration de techniques d'augmentation de données spécifiques aux squelettes
- Optimisation pour l'inférence en temps réel
- Support pour d'autres datasets de squelettes (MHAD, HDM05, etc.)

## Références

Leveraging Pre-trained CNN Models for Skeleton-Based Action Recognition
By Sohaib Laraba(B) , Joelle Tilmanne, and Thierry Dutoit 