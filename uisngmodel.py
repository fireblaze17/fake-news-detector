import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load the trained model
model = joblib.load('D:/Ankush Python Projects/Fake news detector/predictionmodel_sgd.pkl')

# Load the TfidfVectorizer
vectorizer = joblib.load('D:/Ankush Python Projects/Fake news detector/vectorizer.pkl')

# Function to preprocess text
def preprocess_text(text):
    stemmer = PorterStemmer()
    stemmed_content = re.sub('[^a-zA-Z]', ' ', text)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [stemmer.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Function to classify text
def classify_text(text):
    # Preprocess the text
    preprocessed_text = preprocess_text(text)
    # Vectorize the preprocessed text using the loaded vectorizer
    vectorized_text = vectorizer.transform([preprocessed_text])
    # Predict the label
    predicted_label = model.predict(vectorized_text)[0]
    # Map label to human-readable form
    label = "Fake" if predicted_label == 1 else "Real"
    return label

# Example usage
article = input("Enter the article text: ")
result = classify_text(article)
print("The article is classified as:", result)