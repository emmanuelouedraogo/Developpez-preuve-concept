import streamlit as st
import pandas as pd
import os
from PIL import Image, ImageEnhance, ImageFilter
import random
import numpy as np
import urllib.request
import ultralytics
from ultralytics import YOLO
import json
try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False
# Configuration de la page
st.set_page_config(
    page_title="Dashboard Preuve de Concept",
    page_icon="🚗",
    layout="wide"
)

# Titre et Introduction
st.title("🚗 Dashboard de Présentation - Segmentation Sémantique")
st.markdown("""
Bienvenue sur le dashboard de présentation de notre preuve de concept.
Ce projet vise à comparer différentes architectures de réseaux de neurones (U-Net, YOLO) 
pour la segmentation d'images dans un contexte de conduite autonome.
""")

# --- Configuration des chemins ---
# On cherche le dossier 'results' dans le répertoire courant ou aux chemins probables
current_dir = os.path.dirname(os.path.abspath(__file__))
POSSIBLE_PATHS = [
    os.path.join(current_dir, 'results'),
    './results',
    #'/content/drive/MyDrive/P10_Developpez une preuve de concept/results'
    '/results'
]

RESULTS_DIR = None
# On cherche d'abord un dossier qui contient effectivement les images
for path in POSSIBLE_PATHS:
    if os.path.exists(path):
        try:
            if any(f.startswith('comparison_256_') for f in os.listdir(path)):
                RESULTS_DIR = path
                break
        except Exception:
            continue

# Si non trouvé avec les images, on prend le premier chemin valide par défaut
if RESULTS_DIR is None:
    for path in POSSIBLE_PATHS:
        if os.path.exists(path):
            RESULTS_DIR = path
            break

# Si le dossier n'est pas trouvé
if not RESULTS_DIR:
    st.warning("⚠️ Le dossier 'results' est introuvable. Assurez-vous d'avoir exécuté le script de modélisation.")
    # On continue quand même pour afficher l'interface, mais sans données
else:
    st.success(f"Dossier de résultats chargé : `{RESULTS_DIR}`")

CSV_PATH = os.path.join(RESULTS_DIR, 'final_model_comparison_256.csv') if RESULTS_DIR else "final_model_comparison_256.csv"

# --- Section 1: Métriques ---
st.header("1. Comparaison des Performances")

if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    
    # Affichage des données brutes avec mise en forme
    st.subheader("Tableau des résultats")
    st.dataframe(df.style.highlight_max(axis=0, subset=['mIoU'], color='lightgreen')
                 .highlight_min(axis=0, subset=['Inférence (ms)'], color='lightgreen'), 
                 width="stretch")
    
    # Graphiques
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Score mIoU (Plus haut est mieux)")
        if 'Model' in df.columns and 'mIoU' in df.columns:
            chart_data = df.set_index('Model')['mIoU']
            st.bar_chart(chart_data, color="#4CAF50")
        
    with col2:
        st.subheader("Temps d'inférence (Plus bas est mieux)")
        if 'Model' in df.columns and 'Inférence (ms)' in df.columns:
            chart_data_time = df.set_index('Model')['Inférence (ms)']
            st.bar_chart(chart_data_time, color="#FF5722")

    # --- Sous-section: Détails par classe ---
    st.markdown("---")
    st.subheader("Détails par Classe (mAP 50-95)")
    
    class_csv_path = os.path.join(RESULTS_DIR, 'class_performance.csv') if RESULTS_DIR else "class_performance.csv"
    
    # Données fournies
    class_data = {
        'Classe': ['Construction', 'Flat', 'Human', 'Nature', 'Object', 'Sky', 'Vehicle', 'Void'],
        'mAP 50-95': [0.0650, 0.2716, 0.0703, 0.1645, 0.0414, 0.5318, 0.2089, 0.1434]
    }
    
    # Création du CSV si nécessaire
    if not os.path.exists(class_csv_path):
        try:
            pd.DataFrame(class_data).to_csv(class_csv_path, index=False)
        except Exception:
            pass # On utilise les données en mémoire si l'écriture échoue
            
    # Lecture ou utilisation directe
    df_class = pd.read_csv(class_csv_path) if os.path.exists(class_csv_path) else pd.DataFrame(class_data)
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(df_class.style.format({"mAP 50-95": "{:.4f}"}).highlight_max(subset=['mAP 50-95'], color='lightgreen'), use_container_width=True)
    with c2:
        st.bar_chart(df_class.set_index('Classe')['mAP 50-95'], color="#2196F3")

else:
    st.warning(f"Le fichier CSV des résultats ({CSV_PATH}) est manquant. Veuillez lancer l'entraînement.")

# --- Section 2: Visualisation ---
st.header("2. Visualisation des Prédictions")
st.markdown("Comparaison visuelle entre l'image originale et les prédictions des différents modèles.")

if RESULTS_DIR and os.path.exists(RESULTS_DIR):
    # Recherche des images de comparaison générées par le script
    target_images = ["comparison_256_val34.png", "comparison_256_val45.png", "comparison_256_val85.png"]
    
    # On privilégie les images spécifiques demandées si elles existent
    comparison_images = [f for f in target_images if os.path.exists(os.path.join(RESULTS_DIR, f))]
    
    # Sinon, on cherche toutes les images disponibles
    if not comparison_images:
        comparison_images = [f for f in os.listdir(RESULTS_DIR) if f.startswith('comparison_256_') and f.endswith(('.png', '.jpg', '.jpeg'))]

    if comparison_images:
        # Sélecteur d'image
        selected_file = st.selectbox("Choisir une image de test :", comparison_images)
        
        if selected_file:
            img_path = os.path.join(RESULTS_DIR, selected_file)
            image = Image.open(img_path)
            st.image(image, caption=f"Résultats pour {selected_file}", use_column_width=True)
    else:
        st.info("Aucune image de comparaison trouvée dans le dossier results.")
else:
    st.info("Dossier de résultats non accessible pour les images.")

# --- Section 2.5: Exploratory Data Analysis (EDA) ---
st.header("2.5. Exploratory Data Analysis")
st.markdown("Exploring the dataset with sample images and transformations.")

if RESULTS_DIR and os.path.exists(RESULTS_DIR):
    # Get all image files from the directory
    all_images = [f for f in os.listdir(RESULTS_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Filter out comparison images to only show original dataset images
    dataset_images = [f for f in all_images if not f.startswith('comparison_256_')]

    if dataset_images:
        # Sample display of images
        num_samples = min(5, len(dataset_images))
        sample_images = random.sample(dataset_images, num_samples)

        st.subheader("Sample Images from Dataset")
        cols = st.columns(num_samples)
        for i, img_file in enumerate(sample_images):
            img_path = os.path.join(RESULTS_DIR, img_file)
            img = Image.open(img_path)
            cols[i].image(img, use_column_width=True, caption=img_file)

        # Image Transformations (example: equalization and blurring)
        st.subheader("Image Transformations")
        selected_image = st.selectbox("Select an image for transformation:", sample_images)
        image_path = os.path.join(RESULTS_DIR, selected_image)
        img = Image.open(image_path)

        col1, col2 = st.columns(2)

        with col1:
            st.image(img, caption=f"Original: {selected_image}", use_column_width=True)

        with col2:
            # Equalization
            img_eq = ImageEnhance.Contrast(img).enhance(1.5)  # Increase contrast for equalization effect
            st.image(img_eq, caption="Equalized", use_column_width=True)

        col3, col4 = st.columns(2)
        with col3:
             # Blurring
            img_blur = img.filter(ImageFilter.BLUR)
            st.image(img_blur, caption="Blurred", use_column_width=True)
        
        with col4:
            # Convert the image to grayscale
            img_gray = img.convert('L')
            st.image(img_gray, caption="Grayscale", use_column_width=True)


    else:
        st.info("No dataset images found in the results folder.")
else:
    st.info("Results folder not accessible for EDA.")

# --- Fonctions utilitaires pour le modèle Keras (U-Net VGG16) ---
def create_weighted_loss(class_weights):
    class_weights_tensor = tf.constant(class_weights, dtype=tf.float32)
    def weighted_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        pixel_weights = tf.gather(class_weights_tensor, y_true)
        scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')
        unweighted_loss = scce(y_true, y_pred)
        weighted_loss = unweighted_loss * pixel_weights
        return tf.reduce_mean(weighted_loss)
    return weighted_loss

def load_keras_segmentation_model(model_path, config_path, class_weights_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Chargement des poids
    if os.path.exists(class_weights_path):
        with open(class_weights_path, 'r') as f:
            class_weights = json.load(f)
    else:
        class_weights = [1.0] * config.get('num_classes', 8)
        
    custom_objects = {'weighted_loss': create_weighted_loss(class_weights)}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    return model, config

def preprocess_image_keras(image_pil, img_size=(224, 224)):
    image_array = np.array(image_pil)
    original_size = image_array.shape[:2]
    
    image_resized = tf.image.resize(image_array, img_size, method='bilinear')
    image_normalized = tf.cast(image_resized, tf.float32) / 255.0
    image_batch = tf.expand_dims(image_normalized, axis=0)
    
    return image_batch, original_size

def predict_keras(model, processed_image_batch, original_size):
    predictions = model.predict(processed_image_batch, verbose=0)
    pred_mask_resized = tf.argmax(predictions, axis=-1)[0].numpy()
    
    pred_mask_original = tf.image.resize(
        np.expand_dims(pred_mask_resized, axis=-1),
        original_size,
        method='nearest'
    )
    return tf.squeeze(pred_mask_original, axis=-1).numpy()

def create_colored_mask(prediction_mask, config):
    if 'group_colors' not in config:
        return Image.fromarray((prediction_mask * 30).astype(np.uint8)) # Fallback
        
    colors = np.array(config['group_colors'], dtype=np.uint8)
    # Gestion des index hors limites si nécessaire
    num_colors = len(colors)
    mask_safe = np.clip(prediction_mask, 0, num_colors - 1)
    
    rgb_mask = colors[mask_safe]
    return Image.fromarray(rgb_mask.astype(np.uint8))

# --- Section 3: Prédiction Live ---
st.header("3. Prédiction Live (Optionnel)")
st.markdown("Testez les modèles entraînés sur vos propres images.")

live_model_dir = RESULTS_DIR if RESULTS_DIR else "."

# Sélecteur de modèle
model_option = st.selectbox(
    "Choisir le modèle :",
    ["YOLOv9-seg", "U-Net VGG16 (Keras)"]
)

# Upload image
uploaded_file = st.file_uploader("Charger une image pour prédiction", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Image chargée", use_column_width=True)
    
    if st.button("Lancer la prédiction"):
        with st.spinner("Chargement du modèle et inférence..."):
            try:
                if "YOLO" in model_option:
                    # Configuration selon la version choisie
                    model_filename = "final_best_yolov9.pt"
                    model_url = "https://github.com/emmanuelouedraogo/Developpez-preuve-concept/releases/download/v0.1.0/final_best_yolov9.pt"
                    
                    model_path = os.path.join(live_model_dir, model_filename)
                    
                    # Téléchargement automatique si nécessaire
                    if not os.path.exists(model_path) and model_url:
                        st.info(f"Téléchargement du modèle {model_filename} depuis GitHub...")
                        try:
                            urllib.request.urlretrieve(model_url, model_path)
                            st.success("Téléchargement terminé.")
                        except Exception as e:
                            st.error(f"Erreur lors du téléchargement : {e}")
                    
                    # Import dynamique pour éviter les crashs si la lib manque
                    from ultralytics import YOLO
                    
                    if os.path.exists(model_path):
                        model = YOLO(model_path)
                        results = model.predict(source=image, conf=0.25)
                        # plot() retourne un array BGR
                        res_plotted = results[0].plot()
                        st.image(res_plotted, caption=f"Résultat {model_option}", use_column_width=True, channels="BGR")
                    else:
                        st.warning(f"Modèle introuvable à l'emplacement : {model_path}")
                        
                elif model_option == "U-Net VGG16 (Keras)":
                    if not HAS_TF:
                        st.error("TensorFlow n'est pas installé. Veuillez l'installer via `pip install tensorflow` ou l'ajouter au fichier `requirements.txt`.")
                        st.stop()

                    # URLs from segmentation_pipeline.py
                    model_filename = "final_optimized_model.keras"
                    model_url = "https://github.com/emmanuelouedraogo/voiture-autonaume/releases/download/v.0.0.2/final_optimized_model.keras"
                    config_url = "https://github.com/emmanuelouedraogo/voiture-autonaume/releases/download/v.0.0.2/class_mapping.json"
                    weights_url = "https://github.com/emmanuelouedraogo/voiture-autonaume/releases/download/v.0.0.2/class_weights.json"
                    
                    model_path = os.path.join(live_model_dir, model_filename)
                    config_path = os.path.join(live_model_dir, "class_mapping.json")
                    weights_path = os.path.join(live_model_dir, "class_weights.json")
                    
                    # Téléchargement automatique
                    files_to_download = [
                        (model_path, model_url),
                        (config_path, config_url),
                        (weights_path, weights_url)
                    ]
                    
                    for fpath, furl in files_to_download:
                        if not os.path.exists(fpath):
                            st.info(f"Téléchargement de {os.path.basename(fpath)}...")
                            try:
                                urllib.request.urlretrieve(furl, fpath)
                            except Exception as e:
                                st.error(f"Erreur téléchargement {os.path.basename(fpath)}: {e}")
                    
                    if os.path.exists(model_path):
                        # Chargement du modèle Keras
                        model, config = load_keras_segmentation_model(model_path, config_path, weights_path)
                        
                        # Prétraitement
                        processed_batch, original_size = preprocess_image_keras(image, img_size=(224, 224))
                        
                        # Prédiction
                        mask = predict_keras(model, processed_batch, original_size)
                        
                        # Création de l'image colorée
                        mask_img = create_colored_mask(mask, config)
                        mask_img = mask_img.resize(image.size, resample=Image.NEAREST)
                        
                        # Récupération des infos pour la légende
                        group_names = config.get('group_names', [])
                        group_colors = config.get('group_colors', [])
                        
                        # Affichage côte à côte avec superposition
                        col_res1, col_res2 = st.columns(2)
                        
                        with col_res1:
                            st.image(mask_img, caption="Masque de segmentation (U-Net VGG16)", use_column_width=True)
                            
                        with col_res2:
                            # Superposition
                            img_base = image.convert("RGBA")
                            mask_rgba = mask_img.convert("RGBA")
                            mask_rgba.putalpha(128)
                            overlay = Image.alpha_composite(img_base, mask_rgba)
                            st.image(overlay, caption="Superposition", use_column_width=True)
                        
                        # Légende
                        st.markdown("#### Légende")
                        legend_cols = st.columns(4)
                        for i, (name, color) in enumerate(zip(group_names, group_colors)):
                            with legend_cols[i % 4]:
                                color_hex = '#{:02x}{:02x}{:02x}'.format(*color)
                                st.markdown(f"<div style='display:flex;align-items:center;margin-bottom:5px;'><div style='width:20px;height:20px;background-color:{color_hex};margin-right:8px;border:1px solid #ccc;'></div>{name}</div>", unsafe_allow_html=True)
                    else:
                        st.warning(f"Modèle introuvable à l'emplacement : {model_path}")
            
            except ImportError as e:
                st.error(f"Librairie manquante : {e}. Installez torch ou ultralytics.")
            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {e}")

# --- Section 4: Détails Techniques ---
with st.expander("ℹ️ Détails Techniques du Projet"):
    st.markdown("""
    **Dataset :** Cityscapes (adapté)
    **Classes :** 8 classes (flat, human, vehicle, construction, object, nature, sky, void)
    **Taille d'entrée :** 256x256 pixels (YOLO), 224x224 pixels (U-Net VGG16)
    
    **Modèles testés :**
    1.  **U-Net VGG16 (Keras) :** Architecture encoder-decoder avec Backbone VGG16, optimisée.
    2.  **YOLOv8n-seg :** Modèle SOTA pour la segmentation d'instance/sémantique, optimisé pour la vitesse.
    3.  **YOLOv9c-seg :** Version plus récente et plus complexe de YOLO.
    """)

if st.sidebar.button("Rafraîchir les données"):
    st.rerun()
