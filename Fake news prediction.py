import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset
news_dataset = pd.read_csv('D:/Ankush Python Projects/Fake news detector/train.csv')

# Fill missing values
news_dataset = news_dataset.fillna('')

# Define X and Y
X = news_dataset['text'].values
Y = news_dataset['label'].values

# Perform stemming
port_stem = PorterStemmer()
def stemming(text):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', text)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

X_processed = [stemming(text) for text in X]

# Vectorize the text
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X_processed)

# Split the data for training and testing
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X_vectorized, Y, test_size=0.2, stratify=Y, random_state=2)

# Save the training data and the vectorizer using joblib
joblib.dump(Xtrain, 'Xtrain_sparse.pkl')
joblib.dump(Xtest, 'Xtest_sparse.pkl')
joblib.dump(Ytrain, 'Ytrain.pkl')
joblib.dump(Ytest, 'Ytest.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
