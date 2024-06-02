from fastapi.testclient import TestClient
from main import app
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

client = TestClient(app)

def test_prediction():
    # Simuler une requête POST avec des données de formulaire
    response = client.post("/predict", data={"Title": "'django update model field based another field'", "Body": "'new django python want something used often java consider following model relevant class class item model model name model charfield model decimalfield default def self return self name class saledetail model model item model foreignkey item deposit model foreignkey deposit quantity model positiveintegerfield model decimalfield sale model foreignkey sale def self value self none self value want time item added saledetail update saledetail item saledetail new set used java pojos include logic setter method tried using python property encapsulate item property django update field directly hood would break automatic functionallity also tried subclassing foreignkey accept callback function find way call method container class want provide default want include logic view logic since conceptually think logic model server side use case would update total sale detail sale would like calculate user decides save sale save signal would work thanks'"})
    assert response.status_code == 200

    # Vérifier que la réponse contient la prédiction
    assert "Prédiction" in response.text
    

