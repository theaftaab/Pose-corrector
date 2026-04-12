from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import Response

from services.poseService import pose_service

router = APIRouter(prefix="/pose")


@router.post("", response_class=Response)
async def run_pose(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    image_bytes = await image.read()
    annotated_bytes = pose_service.run(image_bytes)

    return Response(content=annotated_bytes, media_type="image/jpeg")
