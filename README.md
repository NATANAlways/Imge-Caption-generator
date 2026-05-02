# Image Caption Generator — Web Application

A full-stack web application that generates natural language captions for images using deep learning. Users can upload any image and choose between two trained models to generate a caption.

---

## Models

| Model | Architecture | BLEU-4 |
|---|---|---|
| **V1** | Custom 5-block CNN + 2-layer LSTM | ~0.07 |
| **V2** | Pretrained ResNet50 + Attention LSTM | ~0.20 |

Both models were trained on the **Flickr8k dataset** (8,091 images, ~5 captions each).

**V2 improvements over V1:**
- ResNet50 pretrained on 1.2M ImageNet images — richer visual features
- Bahdanau attention — focuses on different image regions per word
- Label smoothing — prevents overconfident predictions
- Length-normalised beam search — produces more complete captions

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React + Vite |
| Backend | FastAPI (Python) |
| ML Framework | PyTorch |
| Containerisation | Docker + Docker Compose |
| Web Server | Nginx (frontend in production) |

---

## Project Structure

```
image_caption_app/
├── backend/
│   ├── main.py                  # FastAPI app entry point + model loading
│   ├── api/
│   │   └── routes.py            # POST /api/caption endpoint
│   ├── services/
│   │   ├── caption_service.py   # Model registry + inference logic
│   │   └── model_definitons.py  # V1 and V2 model class definitions
│   ├── models/
│   │   └── schemas.py           # Request/response data schemas
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Main app — state management
│   │   ├── api/
│   │   │   └── captionApi.js    # All API calls in one place
│   │   └── components/
│   │       ├── ImageUploader.jsx
│   │       ├── ModelSelector.jsx
│   │       └── CaptionDisplay.jsx
│   └── package.json
├── checkpoints/
│   └── vocab.json               # Vocabulary mapping (word ↔ index)
├── docker/
│   ├── Dockerfile.backend
│   └── Dockerfile.frontend
└── docker-compose.yml
```

---

## Running Locally

### Prerequisites
- Python 3.10+
- Node.js 20+
- Trained model files: `best_model.pt` (V1) and `best_model_v2.pt` (V2)

Place model files in `checkpoints/`:
```
checkpoints/
├── best_model.pt
├── best_model_v2.pt
└── vocab.json
```

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`

---

## Running with Docker

```bash
docker-compose up --build
```

- Frontend → `http://localhost:3000`
- Backend API → `http://localhost:8000`
- API Docs → `http://localhost:8000/docs`

---

## API

### `POST /api/caption`

Generate a caption for an uploaded image.

**Request**
```
Content-Type: multipart/form-data

file        : image file (jpg, png, webp)
model_name  : "v1" or "v2"  (default: "v2")
```

**Response**
```json
{
  "caption": "a black dog runs across a green field",
  "model":   "v2",
  "time_ms": 245.3
}
```

**Error Response**
```json
{
  "detail": "File must be jpg, png or webp"
}
```

---

## Environment Variables

Create `frontend/.env.local` for local development:
```
VITE_API_URL=http://localhost:8000/api
```

Create `frontend/.env.production` for deployment:
```
VITE_API_URL=https://your-backend-url/api
```

---

## How It Works

```
User uploads image
       │
       ▼
React frontend (ImageUploader)
       │  multipart/form-data POST
       ▼
FastAPI backend (routes.py)
       │  validates file type
       ▼
CaptionService.predict()
       │  PIL → tensor → model.generate()
       ▼
V1 or V2 model (beam search)
       │
       ▼
{ caption, model, time_ms }
       │
       ▼
CaptionDisplay component
```

---

## Notes

- Model files are excluded from git (exceeds GitHub 100MB limit). Download and place them in `checkpoints/` manually.
- `vocab.json` is included in git (158KB).
- Adding a new model: add a `_load_v3()` method in `caption_service.py` and register it in the `models` dict. No other files need to change.
