import io
import os
import sys
import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, UploadFile, File
from PIL import Image

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import HindiCNN

app = FastAPI()

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# LOAD MODEL
# -----------------------------
MODEL_PATH = os.path.join("models", "hindi_cnn_best.pth")

model = HindiCNN(num_classes=46)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# -----------------------------
# IMAGE TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# -----------------------------
# CLASS LABELS
# -----------------------------
classes = [
'character_10_yna','character_11_taamatar','character_12_thaa','character_13_daa',
'character_14_dhaa','character_15_adna','character_16_tabala','character_17_tha',
'character_18_da','character_19_dha','character_1_ka','character_20_na',
'character_21_pa','character_22_pha','character_23_ba','character_24_bha',
'character_25_ma','character_26_yaw','character_27_ra','character_28_la',
'character_29_waw','character_2_kha','character_30_motosaw','character_31_petchiryakha',
'character_32_patalosaw','character_33_ha','character_34_chhya','character_35_tra',
'character_36_gya','character_3_ga','character_4_gha','character_5_kna',
'character_6_cha','character_7_chha','character_8_ja','character_9_jha',
'digit_0','digit_1','digit_2','digit_3','digit_4','digit_5','digit_6',
'digit_7','digit_8','digit_9'
]

# -----------------------------
# CLASS → HINDI CHARACTER MAP
# -----------------------------
label_to_char = {
'character_1_ka': 'क','character_2_kha': 'ख','character_3_ga': 'ग','character_4_gha': 'घ',
'character_5_kna': 'ङ','character_6_cha': 'च','character_7_chha': 'छ','character_8_ja': 'ज',
'character_9_jha': 'झ','character_10_yna': 'ञ','character_11_taamatar': 'ट','character_12_thaa': 'ठ',
'character_13_daa': 'ड','character_14_dhaa': 'ढ','character_15_adna': 'ण','character_16_tabala': 'त',
'character_17_tha': 'थ','character_18_da': 'द','character_19_dha': 'ध','character_20_na': 'न',
'character_21_pa': 'प','character_22_pha': 'फ','character_23_ba': 'ब','character_24_bha': 'भ',
'character_25_ma': 'म','character_26_yaw': 'य','character_27_ra': 'र','character_28_la': 'ल',
'character_29_waw': 'व','character_30_motosaw': 'श','character_31_petchiryakha': 'ष','character_32_patalosaw': 'स',
'character_33_ha': 'ह','character_34_chhya': 'क्ष','character_35_tra': 'त्र','character_36_gya': 'ज्ञ',
'digit_0': '०','digit_1': '१','digit_2': '२','digit_3': '३','digit_4': '४',
'digit_5': '५','digit_6': '६','digit_7': '७','digit_8': '८','digit_9': '९'
}

# -----------------------------
# ROOT ENDPOINT
# -----------------------------
@app.get("/")
def home():
    return {"message": "Hindi Character Recognition API is running"}

# -----------------------------
# PREDICTION ENDPOINT
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("L")

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    pred_class = classes[pred.item()]
    pred_char = label_to_char[pred_class]

    return {
        "class": pred_class,
        "character": pred_char,
        "confidence": float(confidence.item())
    }