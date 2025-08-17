import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import zipfile, os, io
import pandas as pd
from io import StringIO

st.set_page_config(page_title="Diagnostic Phytosanitaire IA", layout="wide")

# ==========================
# CONFIGURATION DU MODÈLE
# ==========================
MODEL_PATH = "Models/best_student_mobilenetv3_model_6.pth"
FICHES_DIR = "Fiches"
IMG_SIZE = 224

# Classes connues (à adapter selon ton dataset)
class_names = sorted([f.replace(".txt", "") for f in os.listdir(FICHES_DIR) if f.endswith(".txt")])

# Charger le modèle
@st.cache_resource
def load_model():
    model = models.mobilenet_v3_small(weights=None)  # architecture
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# Transformations pour prédiction
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==========================
# FONCTIONS UTILES
# ==========================
def predict(image):
    """Retourne la classe prédite et la confiance."""
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    return class_names[pred.item()], conf.item()

def load_fiche(classe):
    """Charge la fiche texte associée à une classe."""
    fiche_path = os.path.join(FICHES_DIR, f"{classe}.txt")
    if os.path.exists(fiche_path):
        with open(fiche_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Aucune fiche disponible."

def handle_upload(uploaded_file):
    """Retourne une liste de tuples (image PIL, nom du fichier)."""
    images = []
    if uploaded_file.type in ["image/jpeg", "image/png"]:
        image = Image.open(uploaded_file).convert("RGB")
        images.append((image, uploaded_file.name))
    elif uploaded_file.type in ["application/x-zip-compressed", "application/zip"]:
        with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
            for file in zip_ref.namelist():
                if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    img_data = zip_ref.read(file)
                    image = Image.open(io.BytesIO(img_data)).convert("RGB")
                    images.append((image, os.path.basename(file)))
    return images

# ==========================
# INTERFACE STREAMLIT
# ==========================

menu = st.sidebar.radio("Navigation", ["📖 Informations", "🔎 Tester le prototype"])

# -------------------------
# PAGE 1 : Informations
# -------------------------
if menu == "📖 Informations":
    st.title("📖 Projet : Diagnostic phytosanitaire sur mobile")
    st.subheader("Auteur : Marie-Ange DIENG")
    
    st.markdown("""
    ### 🎯 Problème à résoudre
    Les maladies des plantes représentent une menace majeure pour la sécurité alimentaire.  
    Ce projet vise à fournir une solution **IA accessible** pour aider les agriculteurs à diagnostiquer rapidement et efficacement les maladies à partir d’images de plantes.

    ### 🧠 Méthodologie
    - Utilisation du **Transfer Learning**  
    - Comparaison de plusieurs architectures légères (MobileNetV2, EfficientNetB0, etc.)  
    - Sélection du modèle le plus adapté pour mobile et web.

    ### 🤖 Modèle retenu
    - **Teacher-Student avec ResNet50/MobileNetV3-Small**  
    - **Précision** : 100%  
    - **Taille** : ~6 Mo (idéal pour mobile)  
    - **Avantages** : rapide, léger, fiable.  

    ### 📱 Résultat attendu
    - Déploiement d’un prototype accessible via une **web app Streamlit**.  
    - L’utilisateur charge une image, l’IA donne le diagnostic instantanément accompagné d’une fiche explicative.  
    """)

# -------------------------
# PAGE 2 : Prototype
# -------------------------
elif menu == "🔎 Tester le prototype":
    st.title("🔎 Tester le prototype de diagnostic")
    st.write("Suivez les étapes ci-dessous :")

    # Étape 1 : Upload
    st.subheader("1️⃣ Charger une image ou un fichier ZIP")
    uploaded_file = st.file_uploader("Sélectionnez une image (.jpg/.png) ou un fichier .zip contenant plusieurs images", type=["jpg", "jpeg", "png", "zip"])

    if uploaded_file:
        images = handle_upload(uploaded_file)
        if not images:
            st.error("⚠️ Aucun fichier image valide trouvé.")
        else:
            st.success(f"{len(images)} image(s) chargée(s).")

            # Étape 2 : Prédiction
            st.subheader("2️⃣ Prédiction...")

            results = []
            for idx, (img, filename) in enumerate(images, 1):
                pred_class, conf = predict(img)
                fiche = load_fiche(pred_class)

                st.markdown(f"### 🖼️ Image {idx}")
                st.image(img, caption=f"Image {idx}", use_container_width=True)

                st.markdown(f"""
                - ✅ Classe prédite : **{pred_class}**  
                - 🔢 Confiance : **{conf*100:.2f}%**
                """)

                with st.expander(f"📄 Fiche d'information - {pred_class}"):
                    st.write(fiche)

                results.append((idx, filename, pred_class, conf))

            # Étape 3 : Résumé
            st.subheader("3️⃣ Résumé des prédictions")
            df_results = pd.DataFrame(results, columns=["Image #", "Nom du fichier","Classe prédite", "Confiance"])
            st.table(df_results)

            # Conversion en CSV (en mémoire)
            csv_buffer = StringIO()
            df_results.to_csv(csv_buffer, index=False)

            # Bouton de téléchargement
            st.download_button(
                label="📥 Télécharger les prédictions en CSV",
                data=csv_buffer.getvalue(),
                file_name="predictions.csv",
                mime="text/csv"
            )