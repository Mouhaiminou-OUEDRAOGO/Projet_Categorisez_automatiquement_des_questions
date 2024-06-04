import streamlit as st
import requests
import os

def get_prediction(title, body):
    url = os.getenv('PREDICTION_API_URL', 'http://127.0.0.1:8000/predict') 
    data = {"Title": title, "Body": body}
    response = requests.post(url, data=data)
    
    if response.status_code == 200:
        try:
            # Extraire la prédiction de la réponse HTML
            start_index = response.text.find("Prédiction:") + len("Prédiction:")
            end_index = response.text.find("</p>", start_index)
            prediction = response.text[start_index:end_index].strip()
            return prediction
        except Exception as e:
            print("Erreur lors de l'extraction de la prédiction :", e)
            return None
    else:
        print("Erreur HTTP :", response.status_code)
        return None


# Interface utilisateur Streamlit
def main():
    st.title("Application de suggestion de Tags")

    # Champs de saisie pour le titre et le  corps
    title = st.text_input("Titre")
    body = st.text_area("Corps")

    # Bouton de prédiction
    if st.button("Prédire"):
        if title and body:
            prediction = get_prediction(title, body)
            st.write(f"Prédiction: {prediction}")
        else:
            st.warning("Veuillez remplir tous les champs.")

if __name__ == "__main__":
    main()