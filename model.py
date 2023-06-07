import nltk
nltk.download('wordnet')
import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

class MultinomialNaiveBayes:
    def __init__(self):
        self.classes = None
        self.class_probs = None
        self.feature_probs = None
        self.vectorizer = None
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def preprocess(self, text):
        # Tokenize the text
        tokens = text.split()
        
        # Apply stemming and lemmatization
        stemmed_tokens = [self.stemmer.stem(token) for token in tokens]
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in stemmed_tokens]
        
        # Join the tokens back into a string
        preprocessed_text = ' '.join(lemmatized_tokens)
        
        return preprocessed_text

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_probs = np.zeros(len(self.classes))
        self.vectorizer = CountVectorizer(preprocessor=self.preprocess)
        X_transformed = self.vectorizer.fit_transform(X)
        self.feature_probs = np.zeros((len(self.classes), X_transformed.shape[1]))

        for i, cls in enumerate(self.classes):
            X_cls = X_transformed[y == cls]
            self.class_probs[i] = len(X_cls) / len(X_transformed)

            total_counts = np.sum(X_cls, axis=0)
            total_words = np.sum(total_counts)
            self.feature_probs[i] = (total_counts + 1) / (total_words + X_transformed.shape[1])

    def predict(self, X):
        X_transformed = self.vectorizer.transform(X)
        preds = []

        for x in X_transformed:
            probs = np.log(self.class_probs) + np.sum(np.log(self.feature_probs) * x, axis=1)
            pred = self.classes[np.argmax(probs)]
            preds.append(pred)

        return preds

# Example usage
data = pd.read_csv('your_dataset.csv')  # Replace 'your_dataset.csv' with the path to your dataset file

X = data['Action'].values
y = data['Product'].values

model = MultinomialNaiveBayes()
model.fit(X, y)

test_X = np.array([
    "Another user action"
])

predictions = model.predict(test_X)
print(predictions)
