from fastapi import FastAPI,UploadFile
from fastapi import File
import numpy as np
import uvicorn
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
import requests

app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8000",
    ]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model=tf.keras.models.load_model("C://Users//Admin//datasets//project-plant_dieases//saved_model//1")
class_name=["Early blight","late blight","Healthy"]
@app.get("/ping")
async def ping():
    return "hello"

def read_file_as_image(data) -> np.ndarray:
   image=  np.array(Image.open(BytesIO(data)))
   return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img1 = np.expand_dims(image,0)
    prediction = model.predict(img1)
    p1 = ""
    index = class_name[np.argmax(prediction[0])]
    confidence=np.max(prediction[0])
    if(index == "Healthy"):
        p1='''It's Healthy, Supply regular  pesticides'''
    elif(index == "Early blight"):
        p1='''Cause: Fungus(Alternaria solani) Prevention: Spraying Fungicide'''
    else:
        p1='''Cause: Pathogen(Phytophthora infestans) Prevention: Destroy cull, or waste, potato tubers'''
    print(index, confidence, p1)
    return{
        "class": index,
        "confidence": float(confidence),
        "response": p1
    }
    
if __name__ == "__main__":
    uvicorn.run(app,host='localhost', port=8000)
