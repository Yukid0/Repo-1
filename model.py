import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import logging

from config import IMAGE_SIZE, BACKBONE, LEARNING_RATE

logger = logging.getLogger(__name__)

def create_model(backbone=BACKBONE, num_classes=None):
    """
    Crée un modèle de classification basé sur un backbone pré-entraîné.
    
    Args:
        backbone (str): Nom du backbone à utiliser ("MobileNetV2", "ResNet50", "EfficientNetB0")
        num_classes (int): Nombre de classes pour la couche de sortie
        
    Returns:
        model: Modèle Keras compilé
        
    Raises:
        ValueError: Si num_classes n'est pas spécifié ou si le backbone n'est pas supporté
    """
    if num_classes is None:
        raise ValueError("Le nombre de classes doit être spécifié")
    
    logger.info(f"Création du modèle avec backbone {backbone} pour {num_classes} classes")
    
    # Couche d'entrée (RGB)
    inputs = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    
    # Sélection du backbone
    try:
        if backbone == "MobileNetV2":
            base_model = MobileNetV2(
                input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
                include_top=False,
                weights='imagenet'
            )
            logger.info("Backbone MobileNetV2 chargé avec succès")
        elif backbone == "ResNet50":
            base_model = ResNet50(
                input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
                include_top=False,
                weights='imagenet'
            )
            logger.info("Backbone ResNet50 chargé avec succès")
        elif backbone == "EfficientNetB0":
            base_model = EfficientNetB0(
                input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
                include_top=False,
                weights='imagenet'
            )
            logger.info("Backbone EfficientNetB0 chargé avec succès")
        else:
            raise ValueError(f"Backbone {backbone} non supporté")
    except Exception as e:
        logger.error(f"Erreur lors du chargement du backbone {backbone}: {e}")
        raise
    
    # Geler le backbone pour le transfer learning
    base_model.trainable = False
    logger.info(f"Backbone gelé pour l'entraînement initial")
    
    # Construire le modèle complet
    try:
        x = base_model(inputs)
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compiler le modèle
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"Modèle créé et compilé avec succès")
        return model
    except Exception as e:
        logger.error(f"Erreur lors de la création du modèle: {e}")
        raise

def unfreeze_layers(model, num_layers=10):
    """
    Dégèle les dernières couches du backbone pour le fine-tuning.
    
    Args:
        model: Modèle Keras
        num_layers: Nombre de couches à dégeler à partir de la fin du backbone
    
    Returns:
        model: Modèle avec les couches dégelées
        
    Raises:
        ValueError: Si le modèle n'a pas de backbone
    """
    try:
        # Trouver le backbone (première couche avec des poids)
        base_model = None
        for layer in model.layers:
            if hasattr(layer, 'layers'):
                base_model = layer
                break
                
        if base_model is None:
            raise ValueError("Impossible de trouver le backbone dans le modèle")
        
        # Dégeler les dernières couches
        base_model.trainable = True
        
        # Geler toutes les couches sauf les dernières
        for layer in base_model.layers[:-num_layers]:
            layer.trainable = False
            
        logger.info(f"Dégelé les {num_layers} dernières couches du backbone pour le fine-tuning")
        
        # Recompiler le modèle avec un taux d'apprentissage plus faible
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE / 10),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    except Exception as e:
        logger.error(f"Erreur lors du dégel des couches: {e}")
        raise 