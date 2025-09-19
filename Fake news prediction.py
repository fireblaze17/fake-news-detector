import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib
import os
import re


current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the dataset
dataset_path = os.path.join(current_dir, 'train.csv')
news_dataset = pd.read_csv(dataset_path)

# Fill missing values
news_dataset = news_dataset.fillna('')

# Define X and Y
X = news_dataset['text'].values
Y = news_dataset['label'].values

# Perform stemming
port_stem = PorterStemmer()
def stemming(text):
    # Remove newline characters
    text = text.replace('\n', ' ')
    # Stemming
    stemmed_content = re.sub('[^a-zA-Z]', ' ', text)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

print('stemming now')   
X_processed = [stemming(text) for text in X]

# Vectorize the text
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X_processed)

# Split the data for training and testing
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X_vectorized, Y, test_size=0.2, stratify=Y, random_state=2)

# Save the training data and the vectorizer using joblib
joblib.dump(Xtrain, os.path.join(current_dir, 'Xtrain_sparse.pkl'))
joblib.dump(Xtest, os.path.join(current_dir, 'Xtest_sparse.pkl'))
joblib.dump(Ytrain, os.path.join(current_dir, 'Ytrain.pkl'))
joblib.dump(Ytest, os.path.join(current_dir, 'Ytest.pkl'))
joblib.dump(vectorizer, os.path.join(current_dir, 'vectorizer.pkl'))





