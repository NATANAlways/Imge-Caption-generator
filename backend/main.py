from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from api.routes import router
from services.caption_service import CaptionService

V1_PATH    = '../checkpoints/best_model.pt'
V2_PATH    = '../checkpoints/best_model_v2.pt'
VOCAB_PATH = '../checkpoints/vocab.json'

# This will hold our single CaptionService instance
caption_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global caption_service
    print('Server starting - loading models...')
    caption_service = CaptionService(V1_PATH, V2_PATH, VOCAB_PATH)
    yield
    print('Server shutting done')

app = FastAPI(title='Image Caption API', version='1.0', lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:3000', 'http://localhost:5173'],
    allow_methods=['*'],
    allow_headers=['*'],
)
app.include_router(router, prefix='/api')