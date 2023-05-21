import cv2
import numpy as np
from anime_face_detector import create_detector
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
from fastapi import FastAPI, File
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])
detector = create_detector('yolov3', device='cpu')


class Box(BaseModel):
    xa: int
    xb: int
    ya: int
    yb: int
    score: float


class KeyPoint(BaseModel):
    x: int
    y: int
    score: float


class Prediction(BaseModel):
    box: Box
    points: list[KeyPoint]


@app.get("/", include_in_schema=False)
async def route_index():
    return RedirectResponse("/docs")


@app.post("/detect-anime-faces", summary="Detect anime faces in an image.")
async def route_detect_anime_faces(file: bytes = File()) -> list[Prediction]:
    array = np.fromstring(file, np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    predictions = detector(image)

    models = []
    for prediction in predictions:
        box = Box(
            xa=prediction['bbox'][0],
            xb=prediction['bbox'][1],
            ya=prediction['bbox'][2],
            yb=prediction['bbox'][3],
            score=prediction['bbox'][4],
        )

        points = []
        for i in range(prediction['keypoints'].shape[0]):
            points.append(KeyPoint(
                x=prediction['keypoints'][i][0],
                y=prediction['keypoints'][i][1],
                score=prediction['keypoints'][i][2],
            ))

        models.append(Prediction(
            box=box,
            points=points,
        ))

    return models
