import io
import torch
import torch.nn as nn
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from torchvision import transforms
from transformers import Dinov2Model

app = FastAPI(
    title="DINOv2 Multi-Series Abnormality Predictor",
    description="2-series (Sagittal + Coronal) knee MRI abnormality predictor using DINOv2",
    version="1.0.0"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------
# Model Architecture (exactly matching your Kaggle training)
# ------------------------------------------------------------------
class DinoV2FeatureExtractor(nn.Module):
    def __init__(self, dino_model):
        super().__init__()
        self.dino = dino_model

    def forward(self, x):
        out = self.dino(x)
        return out.last_hidden_state[:, 0, :]  # CLS token


class MultiSeriesSlice25DModel(nn.Module):
    def __init__(self, num_classes=12, backbone_dim=384, pretrained_backbone=None,
                 freeze_backbone=True, n_series=2):
        super().__init__()
        self.n_series = n_series
        self.backbone = pretrained_backbone

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.attn = nn.Sequential(
            nn.Linear(backbone_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.series_fusion = nn.Sequential(
            nn.Linear(backbone_dim * n_series, backbone_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Linear(backbone_dim, num_classes)

    def encode_one_series(self, volume):
        """
        volume: (B, D, H, W)
        Returns: (B, backbone_dim)
        """
        B, D, H, W = volume.shape
        x = volume.reshape(B * D, 1, H, W)          # (B*D, 1, H, W)
        x = x.repeat(1, 3, 1, 1)                    # (B*D, 3, H, W)
        features = self.backbone(x)                 # (B*D, backbone_dim)
        features = features.view(B, D, -1)          # (B, D, backbone_dim)
        weights = torch.softmax(self.attn(features), dim=1)
        pooled = (features * weights).sum(dim=1)    # (B, backbone_dim)
        return pooled

    def forward(self, volumes):
        """
        volumes: (B, n_series, D, H, W)
        """
        series_features = [
            self.encode_one_series(volumes[:, i]) for i in range(self.n_series)
        ]
        fused = self.series_fusion(torch.cat(series_features, dim=1))
        logits = self.classifier(fused)
        return logits, fused


# ------------------------------------------------------------------
# Load model
# ------------------------------------------------------------------
MODEL_PATH = "best_model_2series.pt"

print("Loading DINOv2 backbone...")
dino_backbone = Dinov2Model.from_pretrained("facebook/dinov2-small")
feature_extractor = DinoV2FeatureExtractor(dino_backbone)

model = MultiSeriesSlice25DModel(
    num_classes=12,
    backbone_dim=384,
    pretrained_backbone=feature_extractor,
    freeze_backbone=True,
    n_series=2,
)

state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()
print(f"Model loaded successfully on {device}")

CLASS_NAMES = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture"
]

# ------------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_slice(contents: bytes) -> torch.Tensor:
    """
    Convert uploaded image to a single-slice tensor of shape (1, H, W)
    """
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = transform(image)          # (3, H, W)
    return tensor[0:1]                 # (1, H, W)  → model will repeat to 3 channels


# ------------------------------------------------------------------
# API Endpoint
# ------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "DINOv2 Knee Abnormality Predictor is running",
        "docs": "/docs",
        "endpoint": "POST /predict (upload series1 + series2)"
    }


@app.post("/predict")
async def predict(
    series1: UploadFile = File(..., description="Sagittal slice"),
    series2: UploadFile = File(..., description="Coronal slice")
):
    # Basic validation
    for f in (series1, series2):
        if f.content_type is None or not f.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail=f"File '{f.filename}' is not a valid image."
            )

    try:
        slice1 = load_slice(await series1.read())   # (1, H, W)
        slice2 = load_slice(await series2.read())   # (1, H, W)

        # Build volume: (B=1, n_series=2, D=1, H, W)
        volumes = torch.stack([slice1, slice2], dim=0).unsqueeze(0).to(device)

        with torch.no_grad():
            logits, _ = model(volumes)
            probs = torch.sigmoid(logits).squeeze().cpu().tolist()

        if isinstance(probs, float):
            probs = [probs]

        return {
            "status": "success",
            "probabilities": {
                name: round(p, 4) for name, p in zip(CLASS_NAMES, probs)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")