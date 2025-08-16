from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credential=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class ComponentData(BaseModel):
    components: dict

@app.post("/process")
async def process_image_and_json(file: UploadFile = File(...), json_data: str = Form(...)):
    """
    Recieve circuit image and connections JSON from frontend and run YOLO
    and LSD detection.
    """

    components = json.load(json_data)
    result = {
        "status": "success",
        "num_components": len(components),
        "filename": file.filename
    }

    return result
