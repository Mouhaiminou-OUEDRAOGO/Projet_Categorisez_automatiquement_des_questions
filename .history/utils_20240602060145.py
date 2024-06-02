
from bs4 import BeautifulSoup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import nltk
nltk.download('stopwords')



def clean_text(text,
               list_rare_words=None,
               min_len_word=3,
               force_is_alpha=True):
    if not list_rare_words:
      list_rare_words = []
    
    # Supprimer les balises HTML
    text = BeautifulSoup(text, "html.parser").get_text()
      
    # Convertir en minuscules
    text = text.lower().strip()
    
    # Tokenisation
    tokenizer = RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(text)
    # Supprimer les stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    #
    non_rare_tokens = [w for w in filtered_tokens if w not in list_rare_words]
    #
    more_than_N = [w for w in non_rare_tokens if len(w) >=min_len_word]
    #
    if force_is_alpha:
      alpha_tokens= [w for w in more_than_N if w.isalpha()]
    # Lemmatisation
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in alpha_tokens]
    # Rejoindre les tokens en texte
    cleaned_text = ' '.join(lemmatized_tokens)
    return cleaned_text



def evaluate_model(pipeline, X_test, y_test, mlb):
    """
    Evaluate the model using the provided pipeline and test data.

    Parameters:
        pipeline: Fitted scikit-learn pipeline containing the model.
        X_test: Test features.
        y_test: True labels for the test set.
        mlb: Fitted label encoder used for transforming labels.

    Returns:
        None
    """
    # Predictions on the test set
    y_pred = pipeline.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Print evaluation metrics
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    