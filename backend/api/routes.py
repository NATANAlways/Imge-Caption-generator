from fastapi import APIRouter, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import main


router = APIRouter()

@router.post('/caption')
async def generate_captions(file: UploadFile, model_name: str = 'v2'):
    if file.content_type not in ('image/jpeg', 'image/png', 'image/webp'):
        raise HTTPException(status_code=400, detail='File must be jpg , png or webp')
    image_bytes = await file.read()
    result = main.caption_service.predict(image_bytes, model_name)
    return JSONResponse(result)

