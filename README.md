### Restaurant Recommendation System Documentation

---

#### Objective:
The main objective of this recommendation system is to encourage outdoor dining trends and analyze sentiments towards particular cuisines and restaurant ratings.

---

#### Importing Libraries:
The necessary libraries for data manipulation, visualization, natural language processing, and machine learning are imported. These include `pandas`, `numpy`, `matplotlib`, `seaborn`, `nltk`, `re`, and scikit-learn modules.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
filterwarnings('ignore')
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
```

---

#### Data Loading:
Two datasets are loaded:
1. `data_names`: Contains restaurant names, links, cost, collections, cuisines, and timings.
2. `data_review`: Contains restaurant reviews, reviewers, ratings, metadata, time, and pictures.

```python
data_names = pd.read_csv('/path/to/Restaurant names and Metadata.csv')
data_review = pd.read_csv('/path/to/Restaurant reviews.csv')
```

---

#### Data Exploration:
Exploratory data analysis is conducted to understand the structure and attributes of the datasets.

```python
print(data_names.shape)
print(data_review.shape)
data_names.sample(5)
data_review.sample(5)
data_names.info()
data_names.nunique()
data_review.info()
data_review.nunique()
```

---

#### Data Merging:
The two datasets are merged into a single dataframe based on the restaurant name.

```python
df = pd.merge(data_names, data_review, how='left', on='Name')
```

---

#### Data Processing:
The merged dataframe is processed to include only the required columns and to handle missing values and data types.

```python
# Load only required columns
required_columns = ['Cuisines', 'Rating', 'Cost', 'Timings']
new_merge_df = pd.read_csv('/path/to/Merge_data.csv', usecols=required_columns)

# Reorder columns
desired_column_order = ['Cuisines', 'Rating', 'Cost', 'Timings']
new_merge_df = new_merge_df[desired_column_order]

# Data Cleaning
df.drop(['Reviewer', 'Time', 'Pictures', 'Links', 'Collections'], axis=1, inplace=True)
df['Cost'] = df['Cost'].apply(str).str.replace(',', '').astype(float)
df['Rating'] = df['Rating'].str.replace('Like', '1').astype(float)
df['Review'] = df['Review'].fillna('-')
```

---

#### Text Preprocessing and Cleaning:
Text data in the 'Review' and 'Cuisines' columns is preprocessed and cleaned for analysis.

```python
# Text Preprocessing
stemmer = PorterStemmer()
df['Review'] = df['Review'].map(lambda x: stem_word(x))
df['Cuisines'] = df['Cuisines'].map(lambda x: stem_word(x))
```

---

#### Restaurant Recommendation System:
A recommendation system is built using TF-IDF vectorization and cosine similarity.

```python
# Building Recommendation System
df.set_index('Name', inplace=True)
indices = pd.Series(df.index)
tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Review'])
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

def recommend(name, cosine_similarities=cosine_similarities):
    # Recommendation function
    ...
    return df_new
```

---

#### Saving the Model:
The trained recommendation model is saved for future use.

```python
import pickle 
filename = 'trained_model.sav'
with open('trained_model.sav', 'wb') as model_file:
    pickle.dump(cosine_similarities, model_file)
```

---

This documentation provides a comprehensive overview of the restaurant recommendation system, including data preprocessing, model development, and model saving steps.
