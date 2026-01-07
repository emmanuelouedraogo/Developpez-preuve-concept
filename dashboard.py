import streamlit as st
import pandas as pd
import os
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import random
import numpy as np
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
POSSIBLE_PATHS = [
    './results',
    os.path.join(os.path.dirname(__file__), 'results'),
    #'/content/drive/MyDrive/P10_Developpez une preuve de concept/results'
    '/results'
]

RESULTS_DIR = None
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

else:
    st.warning(f"Le fichier CSV des résultats ({CSV_PATH}) est manquant. Veuillez lancer l'entraînement.")

# --- Section 2: Visualisation ---
st.header("2. Visualisation des Prédictions")
st.markdown("Comparaison visuelle entre l'image originale et les prédictions des différents modèles.")

if RESULTS_DIR and os.path.exists(RESULTS_DIR):
    # Recherche des images de comparaison générées par le script
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
# --- Section 3: Détails Techniques ---
with st.expander("ℹ️ Détails Techniques du Projet"):
    st.markdown("""
    **Dataset :** Cityscapes (adapté)
    **Classes :** 8 classes (flat, human, vehicle, construction, object, nature, sky, void)
    **Taille d'entrée :** 256x256 pixels
    
    **Modèles testés :**
    1.  **Mini-Unet :** Architecture encoder-decoder classique (Backbone VGG16).
    2.  **YOLOv8n-seg :** Modèle SOTA pour la segmentation d'instance/sémantique, optimisé pour la vitesse.
    3.  **YOLOv9c-seg :** Version plus récente et plus complexe de YOLO.
    """)

if st.sidebar.button("Rafraîchir les données"):
    st.rerun()


from PIL import ImageFilter