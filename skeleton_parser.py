import numpy as np
import re
import cv2
import logging
from config import IMAGE_SIZE

logger = logging.getLogger(__name__)

def parse_skeleton_file(filepath):
    """
    Parse un fichier .skeleton et extrait les coordonnées 3D des joints pour chaque frame.
    
    Args:
        filepath (str): Chemin vers le fichier .skeleton
        
    Returns:
        numpy.ndarray: Tableau de forme (num_frames, num_joints, 3) contenant les coordonnées 3D,
                      ou None en cas d'erreur
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
            # La première ligne contient le nombre de frames
            num_frames = int(lines[0])
            all_frames_data = []
            line_index = 1
            
            # Parcourir chaque frame
            for i in range(num_frames):
                if line_index >= len(lines): 
                    logger.warning(f"Fin de fichier prématurée à la frame {i}/{num_frames}")
                    break
                    
                # Nombre de corps (personnes) dans cette frame
                num_bodies = int(lines[line_index])
                line_index += 1
                
                if num_bodies > 0:
                    # Ignorer l'ID du corps
                    line_index += 1
                    
                    # Nombre de joints dans ce corps
                    num_joints = int(lines[line_index])
                    line_index += 1
                    
                    # Collecter les coordonnées des joints
                    frame_joints = []
                    for j in range(num_joints):
                        if line_index >= len(lines): 
                            logger.warning(f"Fin de fichier prématurée au joint {j}/{num_joints}")
                            break
                            
                        joint_info = lines[line_index].strip().split()
                        line_index += 1
                        
                        # Extraire les coordonnées x, y, z
                        x, y, z = float(joint_info[0]), float(joint_info[1]), float(joint_info[2])
                        frame_joints.append([x, y, z])
                        
                    all_frames_data.append(frame_joints)
                    
                    # Ignorer les autres corps dans cette frame
                    for b in range(num_bodies - 1):
                        line_index += 1  # ID du corps
                        num_joints_other = int(lines[line_index])
                        line_index += (1 + num_joints_other)  # Sauter les joints
                else:
                    # Pas de corps détecté, ajouter une frame vide
                    all_frames_data.append(np.zeros((25, 3)))
                    
            return np.array(all_frames_data)
            
    except Exception as e:
        logger.error(f"Erreur lors du traitement du fichier {filepath}: {e}")
        return None

def sequence_to_image(skeleton_data, target_size=IMAGE_SIZE):
    """
    Convertit une séquence de squelettes 3D en une image RGB avec des lignes horizontales.
    Chaque canal encode un aspect différent des données de squelette:
    - Canal R: Coordonnées X (position horizontale)
    - Canal G: Coordonnées Y (position verticale)
    - Canal B: Coordonnées Z (profondeur)
    
    Cette version transpose explicitement les données pour obtenir des lignes horizontales
    comme dans l'exemple de référence.
    
    Args:
        skeleton_data (numpy.ndarray): Données de squelette de forme (num_frames, num_joints, 3)
        target_size (tuple): Taille cible de l'image (largeur, hauteur)
        
    Returns:
        numpy.ndarray: Image RGB représentant la séquence, ou None en cas d'erreur
    """
    if skeleton_data is None or skeleton_data.shape[0] == 0:
        return None
    
    try:
        # Extraire les coordonnées x, y, z séparément pour chaque frame et joint
        x_data = skeleton_data[:, :, 0]  # (num_frames, num_joints)
        y_data = skeleton_data[:, :, 1]  # (num_frames, num_joints)
        z_data = skeleton_data[:, :, 2]  # (num_frames, num_joints)
        
        # IMPORTANT: Transposer les données pour obtenir des lignes horizontales
        # Maintenant les joints sont sur les lignes et les frames sur les colonnes
        x_data = x_data.T  # (num_joints, num_frames)
        y_data = y_data.T  # (num_joints, num_frames)
        z_data = z_data.T  # (num_joints, num_frames)
        
        # Normaliser chaque canal séparément
        def normalize_channel(channel):
            min_val, max_val = np.min(channel), np.max(channel)
        if max_val - min_val > 0:
                return (channel - min_val) / (max_val - min_val)
            return channel
            
        x_normalized = normalize_channel(x_data)
        y_normalized = normalize_channel(y_data)
        z_normalized = normalize_channel(z_data)
        
        # Convertir en images 8-bit
        r_channel = (x_normalized * 255).astype(np.uint8)
        g_channel = (y_normalized * 255).astype(np.uint8)
        b_channel = (z_normalized * 255).astype(np.uint8)
        
        # Redimensionner chaque canal à la taille cible
        r_resized = cv2.resize(r_channel, target_size, interpolation=cv2.INTER_LINEAR)
        g_resized = cv2.resize(g_channel, target_size, interpolation=cv2.INTER_LINEAR)
        b_resized = cv2.resize(b_channel, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Combiner les canaux pour créer une image RGB
        rgb_image = np.stack([r_resized, g_resized, b_resized], axis=2)
        
        return rgb_image
        
    except Exception as e:
        logger.error(f"Erreur lors de la conversion du squelette en image RGB: {e}")
        return None

def get_label_from_filename(filename):
    """
    Extrait le numéro d'action (label) à partir du nom de fichier.
    
    Args:
        filename (str): Nom du fichier .skeleton
        
    Returns:
        int: Numéro de classe (base 0) ou -1 en cas d'erreur
    """
    try:
        match = re.search(r'A(\d{3})', filename)
        if match:
            return int(match.group(1)) - 1  # Convertir en base 0
        logger.warning(f"Format de nom de fichier non reconnu: {filename}")
        return -1
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction du label depuis {filename}: {e}")
        return -1 