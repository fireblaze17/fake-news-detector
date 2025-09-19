# Fake News Detector

A machine learning project that classifies news articles as "Real" or "Fake" using Natural Language Processing (NLP) techniques and Stochastic Gradient Descent (SGD) classification.

## 📋 Project Overview

This project implements an end-to-end machine learning pipeline to detect fake news articles. It preprocesses text data using NLP techniques, trains a classification model, and provides a simple interface for real-time predictions.

**Key Features:**
- Text preprocessing with stemming and stop word removal
- TF-IDF vectorization for feature extraction
- SGD classifier for binary classification
- Modular design with separate scripts for data preprocessing, training, and prediction
- Persistent model storage for reuse

## 🔧 Technical Implementation

### Architecture
The project follows a modular architecture with three main components:

1. **Data Preprocessing** (`Fake news prediction.py`)
   - Text cleaning and normalization
   - Stemming using Porter Stemmer
   - Stop word removal
   - TF-IDF vectorization

2. **Model Training** (`modeltraining.py`)
   - SGD Classifier with log loss
   - Training/testing split (80/20)
   - Model evaluation and persistence

3. **Prediction Interface** (`uisngmodel.py`)
   - Real-time text classification
   - Interactive command-line interface

### Dataset
- **Size**: 20,800 articles
- **Features**: id, title, author, text, label
- **Labels**: 0 (Real), 1 (Fake)
- **Distribution**: Balanced dataset (10,387 real vs 10,413 fake articles)

### Machine Learning Pipeline

#### 1. Text Preprocessing
```python
# Text normalization and cleaning
text = text.replace('\n', ' ')
stemmed_content = re.sub('[^a-zA-Z]', ' ', text)
stemmed_content = stemmed_content.lower()

# Stemming and stop word removal
stemmed_content = [porter_stem.stem(word) for word in stemmed_content 
                  if word not in stopwords.words('english')]
```

#### 2. Feature Extraction
- **TF-IDF Vectorization**: Converts text to numerical features
- **Sparse Matrix Storage**: Efficient storage for high-dimensional data

#### 3. Model Training
- **Algorithm**: Stochastic Gradient Descent (SGD) Classifier
- **Loss Function**: Log loss (logistic regression)
- **Validation**: Stratified train-test split

## 🚀 Getting Started

### Prerequisites
```bash
pip install nltk pandas scikit-learn joblib
```

### Installation & Setup
1. Clone or download the project files
2. Ensure `train.csv` is in the project directory
3. Download required NLTK data:
```python
import nltk
nltk.download('stopwords')
```

### Usage

#### 1. Data Preprocessing
```bash
python "Fake news prediction.py"
```
This creates preprocessed data files and TF-IDF vectorizer.

#### 2. Model Training
```bash
python modeltraining.py
```
Trains the SGD classifier and saves the model.

#### 3. Make Predictions
```bash
python uisngmodel.py
```
Interactive interface for classifying news articles.

### Example Usage
```bash
$ python uisngmodel.py
Enter the article text: Scientists have discovered that the earth is actually flat...
The article is classified as: Fake
```

## 📊 Model Performance

The SGD classifier achieves high accuracy on both training and testing datasets. The model uses:
- **Training/Testing Split**: 80/20
- **Stratified Sampling**: Maintains label distribution
- **Cross-validation**: Built-in model evaluation

## 📁 Project Structure
```
fake-news-detector/
├── Fake news prediction.py    # Data preprocessing pipeline
├── modeltraining.py           # Model training script
├── uisngmodel.py             # Prediction interface
├── train.csv                 # Dataset
├── predictionmodel_sgd.pkl   # Trained model
├── vectorizer.pkl            # TF-IDF vectorizer
├── Xtrain_sparse.pkl         # Training features
├── Xtest_sparse.pkl          # Testing features
├── Ytrain.pkl                # Training labels
└── Ytest.pkl                 # Testing labels
```

## 🛠️ Technical Skills Demonstrated

### Machine Learning
- **Supervised Learning**: Binary classification
- **Feature Engineering**: TF-IDF vectorization
- **Model Selection**: SGD classifier optimization
- **Model Persistence**: Joblib serialization

### Natural Language Processing
- **Text Preprocessing**: Cleaning, normalization
- **Stemming**: Porter Stemmer implementation
- **Stop Word Removal**: English language processing
- **Tokenization**: Text to feature conversion

### Software Engineering
- **Modular Design**: Separation of concerns
- **File Management**: Automated path handling
- **Error Handling**: Robust data processing
- **Documentation**: Clear, maintainable code

### Data Science
- **Exploratory Data Analysis**: Dataset understanding
- **Data Validation**: Missing value handling
- **Statistical Analysis**: Label distribution analysis
- **Performance Metrics**: Accuracy evaluation

## 🔮 Future Enhancements

- **Model Improvements**: Test additional algorithms (Random Forest, Neural Networks)
- **Feature Engineering**: Add n-grams, word embeddings
- **Web Interface**: Flask/Django deployment
- **Real-time Processing**: API for bulk predictions
- **Performance Metrics**: Precision, recall, F1-score analysis

## 🎯 Business Impact

This project demonstrates practical applications in:
- **Content Moderation**: Social media platforms
- **Journalism**: News verification systems
- **Education**: Media literacy tools
- **Research**: Misinformation studies

---

*This project showcases end-to-end machine learning development, from data preprocessing to model deployment, with a focus on practical NLP applications and software engineering best practices.*