import io
import torch
import torch.nn as nn
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from torchvision import transforms
from transformers import Dinov2Model
from fastapi.responses import HTMLResponse

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
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Knee MRI Abnormality Predictor</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', system-ui, sans-serif;
                background: linear-gradient(135deg, #0f172a, #1e293b);
                color: #e2e8f0;
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }
            .card {
                background: #1e293b;
                border: 1px solid #334155;
                border-radius: 16px;
                padding: 40px;
                max-width: 680px;
                width: 100%;
                box-shadow: 0 20px 40px rgba(0,0,0,0.3);
            }
            h1 {
                font-size: 1.8rem;
                margin-bottom: 8px;
                color: #38bdf8;
            }
            .subtitle {
                color: #94a3b8;
                margin-bottom: 30px;
                font-size: 1.05rem;
            }
            .badge {
                display: inline-block;
                background: #0ea5e9;
                color: white;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 0.8rem;
                margin-bottom: 20px;
            }
            p {
                line-height: 1.6;
                margin-bottom: 16px;
                color: #cbd5e1;
            }
            .endpoints {
                background: #0f172a;
                border-radius: 10px;
                padding: 16px 20px;
                margin: 24px 0;
                font-family: monospace;
            }
            .endpoints a {
                color: #38bdf8;
                text-decoration: none;
            }
            .endpoints a:hover { text-decoration: underline; }
            .btn {
                display: inline-block;
                background: #0ea5e9;
                color: white;
                padding: 12px 24px;
                border-radius: 8px;
                text-decoration: none;
                font-weight: 600;
                margin-top: 10px;
                transition: background 0.2s;
            }
            .btn:hover { background: #0284c7; }
            .footer {
                margin-top: 30px;
                font-size: 0.85rem;
                color: #64748b;
            }
        </style>
    </head>
    <body>
        <div class="card">
            <div class="badge">Live Demo</div>
            <h1>Knee MRI Abnormality Predictor</h1>
            <p class="subtitle">DINOv2-based Multi-Series Model (Sagittal + Coronal)</p>

            <p>
                This API predicts the probability of 12 common knee abnormalities from MRI slices 
                using a fine-tuned DINOv2 vision transformer with multi-series fusion.
            </p>

            <div class="endpoints">
                <div>Interactive Docs → <a href="/docs">/docs</a></div>
                <div>Prediction Endpoint → <a href="/docs#/default/predict_predict_post">POST /predict</a></div>
            </div>

            <p><strong>How to use:</strong></p>
            <p>
                1. Go to <a href="/docs" style="color:#38bdf8">/docs</a><br>
                2. Upload one <strong>Sagittal</strong> and one <strong>Coronal</strong> MRI slice<br>
                3. Get probabilities for ACL, Meniscus tears, OA, Effusion, etc.
            </p>

            <a href="/docs" class="btn">Try the API →</a>

            <div class="footer">
                Built with FastAPI + PyTorch + DINOv2 · Deployed on Render
            </div>
        </div>
    </body>
    </html>
    """


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