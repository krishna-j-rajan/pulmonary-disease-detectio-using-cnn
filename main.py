from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import aiofiles
import os

from predict import getPrediction


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/home")
def getHomePage(request: Request):
    return templates.TemplateResponse("prediction_input.html", {"request": request})


@app.post("/getPrediction", response_class=HTMLResponse)
async def getUploadImagePrediction(request: Request, file: UploadFile):
    async with aiofiles.open(file.filename, 'wb') as out_file:
        content = await file.read()  # async read
        await out_file.write(content)  # async write
    prediction = getPrediction(file.filename)
    print(file.filename, prediction)
    os.remove(file.filename)
    return templates.TemplateResponse("prediction_result.html", {"request": request, "class_num": prediction["class_num"],
                                                          "class_name": prediction["class_name"]})
