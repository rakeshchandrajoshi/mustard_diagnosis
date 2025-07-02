import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import contextlib

# ====== Configuration ======
IMAGE_SIZE = (512, 512)
NUM_CLASSES = 6
MODEL_PATH = 'best_finegrained_model2.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = torch.cuda.is_available()
amp_autocast = torch.cuda.amp.autocast if USE_AMP else contextlib.nullcontext

# ====== Friendly Class Names ======
class_names = [
    "Aphid Infestation: 101–250 Aphids",
    "Aphid Infestation: 20–30 Aphids",
    "Aphid Infestation: 251–500 Aphids",
    "Aphid Infestation: 31–100 Aphids",
    "Aphid Infestation: More Than 500 Aphids",
    "Healthy Mustard Plant"
]

# ====== Technical Labels to Remedies ======
original_labels = [
    'aphid_101_250',
    'aphid_20_30',
    'aphid_251_500',
    'aphid_31_100',
    'aphid_more_than_500',
    'mustard_healthy'
]

remedies = {
    'aphid_20_30': "🟡 **Mild Infestation**: Monitor closely. Natural predators like ladybugs can help.",
    'aphid_31_100': "🟠 **Moderate Infestation**: Use yellow sticky traps and encourage natural predators.",
    'aphid_101_250': "🔴 **High Infestation**: Apply neem oil or insecticidal soap. Monitor for spread.",
    'aphid_251_500': "🔴 **Severe Infestation**: Use strong bio-pesticides or selective chemical treatments.",
    'aphid_more_than_500': "🔴❗ **Very Severe Infestation**: Immediate chemical treatment is recommended.",
    'mustard_healthy': "✅ Your mustard plant is healthy. Keep monitoring for early signs of aphids."
}

# ====== CBAM Module ======
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_attention(x) * x
        sa = self.spatial_attention(torch.cat(
            [ca.mean(1, keepdim=True), ca.max(1, keepdim=True)[0]], dim=1))
        return sa * ca

# ====== FineGrainedNet ======
class FineGrainedNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
        self.cbam = CBAM(2048)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.cbam(x)
        x = self.head(x)
        return x

# ====== Load Model ======
@st.cache_resource
def load_model():
    model = FineGrainedNet(NUM_CLASSES).to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# ====== Image Preprocessing ======
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# ====== Streamlit UI ======
st.title("🌿 Mustard Aphid Condition Classifier")
st.markdown("Upload a mustard plant image to detect **aphid infestation severity** or **healthy condition**.")

uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = preprocess_image(image).to(DEVICE)
    model = load_model()

    with torch.no_grad(), amp_autocast():
        output = model(input_tensor)
        pred_idx = output.argmax(1).item()

    pretty_class = class_names[pred_idx]
    remedy = remedies[original_labels[pred_idx]]

    st.markdown("---")
    st.markdown(f"### 🧪 **Diagnosis**: `{pretty_class}`")
    st.markdown(f"💡 **Recommendation**: {remedy}")





# import streamlit as st
# import torch
# import torch.nn as nn
# from torchvision import transforms, models
# from PIL import Image
# import numpy as np

# # ========= Config ==========
# NUM_CLASSES = 6
# IMAGE_SIZE = (512, 512)
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# SAVE_PATH = 'best_finegrained_model2.pth'

# # ========= Original Class Names ==========
# class_names = ['aphid_101_250', 'aphid_20_30', 'aphid_251_500',
#                'aphid_31_100', 'aphid_more_than_500', 'mustard_healthy']

# # ========= Remedies Dictionary ==========
# remedies = {
#     'aphid_101_250': 'Apply neem oil spray and monitor for changes in 2-3 days.',
#     'aphid_20_30': 'Low infestation. Monitor weekly, no immediate action needed.',
#     'aphid_251_500': 'Consider insecticidal soap or horticultural oil spray.',
#     'aphid_31_100': 'Use yellow sticky traps and encourage natural predators.',
#     'aphid_more_than_500': 'Severe infestation. Immediate use of systemic insecticide recommended.',
#     'mustard_healthy': 'No issues detected. Maintain current farming practices.'
# }

# # ========= CBAM ==========
# class CBAM(nn.Module):
#     def __init__(self, channels, reduction=16, kernel_size=7):
#         super().__init__()
#         self.channel_attention = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channels, channels // reduction, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(channels // reduction, channels, 1, bias=False),
#             nn.Sigmoid()
#         )
#         self.spatial_attention = nn.Sequential(
#             nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         ca = self.channel_attention(x) * x
#         sa = self.spatial_attention(torch.cat(
#             [ca.mean(1, keepdim=True), ca.max(1, keepdim=True)[0]], dim=1))
#         return sa * ca

# # ========= Model Definition ==========
# class FineGrainedNet(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#         backbone = models.resnet50(pretrained=True)
#         self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
#         self.cbam = CBAM(2048)
#         self.head = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.Dropout(0.5),
#             nn.Linear(2048, num_classes)
#         )

#     def forward(self, x):
#         x = self.feature_extractor(x)
#         x = self.cbam(x)
#         x = self.head(x)
#         return x

# # ========= Load Model ==========
# @st.cache_resource
# def load_model():
#     model = FineGrainedNet(NUM_CLASSES).to(DEVICE)
#     model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
#     model.eval()
#     return model

# model = load_model()

# # ========= Transform ==========
# transform = transforms.Compose([
#     transforms.Resize(IMAGE_SIZE),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])
# ])

# # ========= Streamlit UI ==========
# st.title("🌿 Mustard Leaf Condition Classifier")
# st.write("Upload a mustard plant leaf image to diagnose aphid infestation level or health status.")

# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file:
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     # Preprocess and predict
#     input_tensor = transform(image).unsqueeze(0).to(DEVICE)
#     with torch.no_grad():
#         outputs = model(input_tensor)
#         _, predicted = torch.max(outputs, 1)
#         class_index = predicted.item()
#         predicted_class = class_names[class_index]
#         recommendation = remedies[predicted_class]

#     # Display result
#     st.markdown(f"### 🧪 Diagnosis: `{predicted_class}`")
#     st.markdown(f"### 💡 Recommendation: {recommendation}")
