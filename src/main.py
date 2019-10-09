from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import *
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from texttable import Texttable
import pandas as pd

def loadDataFromPath(path):
    df = pd.read_json(path_or_buf=path, lines=True, encoding='utf-8')

    return df

# DATASET INGLESE
# df = loadDataFromPath("../data/Sarcasm_Headlines_Dataset_v2.json")

# DATASET ITALIANO
df = loadDataFromPath("../data/italian_headline.json")
df_not_sarcatic = df[df.is_sarcastic == 0]
df_sarcatic = df[df.is_sarcastic == 1]
df = pd.concat([df_sarcatic, df_not_sarcatic[:10222]])

X_train, X_test, y_train, y_test = train_test_split(df['headline'], df['is_sarcastic'], test_size=0.33, random_state=42)

# pipeline
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

parameters = [{
        'clf': [MultinomialNB()],
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'vect__max_df': np.arange(0.1, 1.0, 0.1),
        'vect__min_df': np.arange(1, 5, 1),
        'tfidf__use_idf': (True, False),
        'tfidf__sublinear_tf': (True, False),
        'tfidf__smooth_idf': (True, False),
        'tfidf__norm': ('l1', 'l2'),
        'clf__alpha': (1e-2, 1e-3),
    },
    {
        'clf': [ComplementNB()],
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'vect__max_df': np.arange(0.1, 1.0, 0.1),
        'vect__min_df': np.arange(1, 5, 1),
        'tfidf__use_idf': (True, False),
        'tfidf__sublinear_tf': (True, False),
        'tfidf__smooth_idf': (True, False),
        'tfidf__norm': ('l1', 'l2'),
        'clf__alpha': (1e-2, 1e-3),
        'clf__norm': ('l1', 'l2')
    },
    {
        'clf': [BernoulliNB()],
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'vect__max_df': np.arange(0.1, 1.0, 0.1),
        'vect__min_df': np.arange(1, 5, 1),
        'tfidf__use_idf': (True, False),
        'tfidf__sublinear_tf': (True, False),
        'tfidf__smooth_idf': (True, False),
        'tfidf__norm': ('l1', 'l2'),
        'clf__alpha': (1e-2, 1e-3),
        'clf__binarize': (0.0, 1.0),
    },
    {
        'clf': [SGDClassifier()],
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'vect__max_df': np.arange(0.1, 1.0, 0.1),
        'vect__min_df': np.arange(1, 5, 1),
        'tfidf__use_idf': (True, False),
        'tfidf__sublinear_tf': (True, False),
        'tfidf__smooth_idf': (True, False),
        'tfidf__norm': ('l1', 'l2'),
        'clf__loss': ('hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'),
        'clf__alpha': (1e-2, 1e-3),
        'clf__penalty': ('l1', 'l2'),
        'clf__l1_ratio': np.arange(0.01, 0.5, 0.05),
        'clf__fit_intercept': (True, False),
        'clf__max_iter': (500, 1000),
        'clf__tol': (1e-2, 1e-3),
        'clf__shuffle': (True, False),
        'clf__eta0': np.arange(0.0, 1.0, 0.1),
        'clf__power_t': np.arange(0.0, 1.0, 0.1),
        'clf__early_stopping': (True, False),
        'clf__validation_fraction': np.arange(0.01, 1.0, 0.1),
        'clf__n_iter_no_change': np.arange(2, 8, 1),
        'clf__average': (True, False)
    },
    {
        'clf': [SVC(gamma='auto')],
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'vect__max_df': np.arange(0.1, 1.0, 0.1),
        'vect__min_df': np.arange(1, 5, 1),
        'tfidf__use_idf': (True, False),
        'tfidf__sublinear_tf': (True, False),
        'tfidf__smooth_idf': (True, False),
        'tfidf__norm': ('l1', 'l2'),
        'clf__alpha': (1e-2, 1e-3),
    },
    {
        'clf': [KMeans(n_clusters=2)],
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'vect__max_df': np.arange(0.1, 1.0, 0.1),
        'vect__min_df': np.arange(1, 5, 1),
        'tfidf__use_idf': (True, False),
        'tfidf__sublinear_tf': (True, False),
        'tfidf__smooth_idf': (True, False),
        'tfidf__norm': ('l1', 'l2'),
        'clf__alpha': (1e-2, 1e-3),
    },
    {
        'clf': [DecisionTreeClassifier()],
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'vect__max_df': np.arange(0.1, 1.0, 0.1),
        'vect__min_df': np.arange(1, 5, 1),
        'tfidf__use_idf': (True, False),
        'tfidf__sublinear_tf': (True, False),
        'tfidf__smooth_idf': (True, False),
        'tfidf__norm': ('l1', 'l2'),
        'clf__alpha': (1e-2, 1e-3),
    }
]

gs_clf = GridSearchCV(text_clf, parameters, cv=5, iid=False, n_jobs=-1)

gs_clf = gs_clf.fit(X_train, y_train)

# test
predict = gs_clf.predict(X_test)

print(gs_clf.best_estimator_._final_estimator,  np.mean(predict == y_test))

print(metrics.classification_report(y_test, predict))
confusion_matrix = metrics.confusion_matrix(y_test, predict)
t = Texttable()
t.add_rows([["", "Is not Sarcastic (PREDICTED)", "Is Sarcastic (PREDICTED)", "TOTAL"],
            ["Is not Sarcastic (REAL)", confusion_matrix[0][0], confusion_matrix[0][1], confusion_matrix[0][0] + confusion_matrix[0][1]],
              ["Is Sarcastic (REAL)", confusion_matrix[1][0], confusion_matrix[1][1], confusion_matrix[1][0] + confusion_matrix[1][1]],
              ["TOTAL", confusion_matrix[0][0] + confusion_matrix[1][0], confusion_matrix[0][1] + confusion_matrix[1][1],
              confusion_matrix[0][0] + confusion_matrix[1][0] + confusion_matrix[0][1] + confusion_matrix[1][1]]])
print(t.draw())