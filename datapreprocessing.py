import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import  OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_iris
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import  Normalizer
from sklearn.preprocessing import  MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer





#"============================================================================================="
                                " label encoder "
#dataset loading
df = pd.read_csv('Data.csv')
df.head()

#dataset cleaning
imputer = Imputer(missing_values='NaN', strategy='mean', axis = 0)
df.iloc[:, 1:3] = imputer.fit_transform(df.iloc[:, 1:3])
df.head()


#loading labelencoder 
lable_encoder = LabelEncoder()
temp = df.copy()
temp.iloc[:, 0] = lable_encoder.fit_transform(df.iloc[:, 0])
temp.head()


#"============================================================================================="
                                   "threshold"
#iris dataset loading
iris_dataset = load_iris()
df[]
X = iris_dataset.data
y = iris_dataset.target
feature_names = iris_dataset.feature_names

#thresholding te dataset
X[:, 1:2] = Binarizer(threshold=X[:, 1].mean()).fit_transform(X[:, 1].reshape(-1, 1))
X[:, 1]


#"============================================================================================"
                                      "scaling Normalizing"

X = df[["value1","value2"]].values.astype(np.float64)
standard_scaler = StandardScaler()
normalizer = Normalizer()
min_max_scaler = MinMaxScaler()


print("Standardization")
print(standard_scaler.fit_transform(X))

print("Normalizing")
print(normalizer.fit_transform(X))

print("MinMax Scaling")
print(min_max_scaler.fit_transform(X))

#"============================================================================================="
                                         "Feature Extraction"

docs = ["Mayur is a nice boy.", "Mayur rock! wohooo!", "My name is Mayur, and I am a Pythonista!"]
cv = CountVectorizer()
X = cv.fit_transform(docs)
print(X.todense())
print(cv.vocabulary_)


docs = [{"Mayur": 1, "is": 1, "awesome": 2}, {"No": 1, "I": 1, "dont": 2, "wanna": 3, "fall": 1, "in": 2, "love": 3}]
dv = DictVectorizer()
X = dv.fit_transform(docs)
print(X.todense())


tfidf_vectorizer = TfidfVectorizer()
cv_vectorizer = CountVectorizer()
docs = ["Mayur is a Guitarist", "Mayur is Musician", "Mayur is also a programmer"]
X_idf = tfidf_vectorizer.fit_transform(docs)
X_cv = cv_vectorizer.fit_transform(docs)
print(X_idf.todense())
print(tfidf_vectorizer.vocabulary_)
print(X_cv.todense())


#|==============================================================================================================================