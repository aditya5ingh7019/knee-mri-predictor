# Knee MRI Abnormality Predictor

**Live Demo:** [https://knee-mri-predictor.onrender.com](https://knee-mri-predictor.onrender.com)

A production-ready AI API that predicts the probability of 12 common knee abnormalities from MRI slices using a fine-tuned **DINOv2** vision transformer with multi-series fusion (Sagittal + Coronal).

---

## Features

- Multi-series input (Sagittal + Coronal MRI slices)
- 12 abnormality predictions with confidence scores
- Built with **FastAPI** + **PyTorch** + **Transformers**
- Fully containerized with **Docker**
- Deployed on **Render** (public live endpoint)

### Predicted Abnormalities

- ACL Tear
- MCL Tear
- Medial Meniscus
- Lateral Meniscus
- Medial OA
- Lateral OA
- PF OA
- Effusion
- Synovitis
- Baker's Cyst
- Contusion
- Fracture

---

## How to Use the Live API

1. Open the interactive docs:  
   → [https://knee-mri-predictor.onrender.com/docs](https://knee-mri-predictor.onrender.com/docs)

2. Use the `/predict` endpoint  
3. Upload two images:
   - `series1` → Sagittal slice
   - `series2` → Coronal slice

4. Receive JSON response with probabilities.

---

## Tech Stack

- **Model**: DINOv2-small + custom Multi-Series Slice 2.5D architecture
- **Framework**: FastAPI
- **Deep Learning**: PyTorch + Hugging Face Transformers
- **Containerization**: Docker
- **Deployment**: Render

---

## Local Development

```bash
# Clone the repository
git clone https://github.com/aditya5ingh7019/knee-mri-predictor.git
cd knee-mri-predictor

```
---

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

---

# Install dependencies
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

---

# Run the API
uvicorn main:app --reload
Open → http://127.0.0.1:8000/docs

---

# Docker
Bashdocker build -t knee-mri-predictor .
docker run -p 8000:8000 knee-mri-predictor

---

# Model Details

Backbone: facebook/dinov2-small
Architecture: Multi-Series Slice 2.5D with attention pooling
Input: 2 series (currently single-slice demo version)
Output: 12-class multi-label probabilities (sigmoid)

---

# Author
Aditya Singh

GitHub: aditya5ingh7019
text---

### After creating the file:

```powershell
git add README.md
git commit -m "Add professional README"
git push
