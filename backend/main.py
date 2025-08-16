from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.classes import run_algo
import json

app = FastAPI()

origins = [
    "http://localhost:5173",  
    "http://127.0.0.1:5173",
    "https://bread-maker-frontend-1b02mgpcd-cadens-projects-d48b21c6.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ComponentData(BaseModel):
    components: dict

@app.post("/process")
async def process_image_and_json(file: UploadFile = File(...), json_data: str = Form(...)):
    """
    Recieve circuit image and connections JSON from frontend and run YOLO
    and LSD detection.
    """

    components = json.loads(json_data)
    contents = await file.read()

    result = run_algo(contents)
    
    return {"status": "success", "components": result["componentIds"], 
            "connections": result["connections"], "lines": result["lines"], "filename": file.filename, 
            "time": result["time"]}
