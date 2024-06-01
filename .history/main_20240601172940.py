from joblib import load
from fastapi import FastAPI, Form
from pydantic import BaseModel
from utils import clean_text
from fastapi.responses import HTMLResponse

app = FastAPI()

model = load('model.classifier')
encoder = load('mlb_encoder')
vectorizer = load('model_vectorizer')

class RequestBody(BaseModel):
    Title: str
    Body: str

@app.get("/", response_class=HTMLResponse)
async def home():
    """ Endpoint racine qui fournit un message de bienvenue """
    html_content = """
    <html>
        <head>
            <title>Accueil</title>
        </head>
        <body>
            <h1>Bienvenue sur votre application de suggestion de Tags</h1>
            <form action="/predict" method="post">
                <label for="title">Titre:</label><br>
                <input type="text" id="title" name="Title"><br>
                <label for="body">Corps:</label><br>
                <textarea id="body" name="Body"></textarea><br>
                <input type="submit" value="Prédire">
            </form>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/predict", response_class=HTMLResponse)
async def predict(Title: str = Form(...), Body: str = Form(...)):
    clean_title = clean_text(Title)
    clean_body = clean_text(Body)
    input_clean = clean_title + " " + clean_body
    X = vectorizer.transform([input_clean])
    result = model.predict(X)
    prediction = encoder.inverse_transform(result)
    
    html_content = f"""
    <html>
        <head>
            <title>Résultat de la prédiction</title>
        </head>
        <body>
            <h1>Résultat de la prédiction</h1>
            <p>Prédiction: {prediction}</p>
            <a href="/">Retour à l'accueil</a>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)