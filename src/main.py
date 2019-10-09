from datacollector import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


# DATASET INGLESE
# df = loadDataFromPath("data/Sarcasm_Headlines_Dataset_v2.json")

# DATASET ITALIANO
df = loadDataFromPath("data/italian_headline.json")

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

parameters = {
    'vect__ngram_range': [(1, 1), (1, 2)],
    # 'vect__max_df': (0.1, 1.0),
    # 'vect__min_df': (1, 5),
    'tfidf__use_idf': (True, False),
    # 'tfidf__sublinear_tf': (True, False),
    # 'tfidf__smooth_idf': (True, False),
    # 'tfidf__norm': ('l1', 'l2'),
    'clf__alpha': (1e-2, 1e-3),
}

gs_clf = GridSearchCV(text_clf, parameters, cv=5, iid=False, n_jobs=-1)

gs_clf = gs_clf.fit(X_train, y_train)

# test
predict = gs_clf.predict(X_test)

print(np.mean(predict == y_test))

print(metrics.classification_report(y_test, predict))

print(metrics.confusion_matrix(y_test, predict))