import base64

from fastapi import APIRouter, File, HTTPException, UploadFile

from models import PersonAnalysis, PostureAnalysisResponse, PostureMetric
from services.poseService import pose_service

router = APIRouter(prefix="/pose")


@router.post("", response_model=PostureAnalysisResponse)
async def run_pose(image: UploadFile = File(...)):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    image_bytes = await image.read()
    analyses, annotated_bytes = pose_service.run(image_bytes)

    people = [
        PersonAnalysis(
            person_id=i,
            view=a.view,
            person_detected=a.person_detected,
            metrics=[
                PostureMetric(
                    region=m.region,
                    value=m.value,
                    unit=m.unit,
                    target=m.target,
                    status=m.status,
                )
                for m in a.metrics
            ],
        )
        for i, a in enumerate(analyses)
    ]

    return PostureAnalysisResponse(
        people=people,
        image_base64=base64.b64encode(annotated_bytes).decode("ascii"),
    )
