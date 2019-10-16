from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import make_scorer, accuracy_score, f1_score
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
import re
import pickle
import matplotlib.pyplot as plt

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
        # ALL PARAMS READY TO RUN 0
        'clf': [MultinomialNB()],
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'vect__max_df': np.arange(0.1, 1.1, 0.1),
        'vect__min_df': np.arange(0, 6, 1),
        'tfidf__use_idf': (True, False),
        # 'tfidf__sublinear_tf': (True, False),
        # 'tfidf__smooth_idf': (True, False),
        # 'tfidf__norm': ('l1', 'l2'),
        'clf__alpha': (1e-2, 1e-3),
    },
    {
        # ALL PARAMS READY TO RUN 1
        'clf': [ComplementNB()],
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'vect__max_df': np.arange(0.1, 1.1, 0.1),
        'vect__min_df': np.arange(0, 6, 1),
        'tfidf__use_idf': (True, False),
        # 'tfidf__sublinear_tf': (True, False),
        # 'tfidf__smooth_idf': (True, False),
        # 'tfidf__norm': ('l1', 'l2'),
        'clf__alpha': (1e-2, 1e-3),
        # 'clf__norm': ('l1', 'l2')
    },
    {
        # ALL PARAMS READY TO RUN 2
        'clf': [BernoulliNB()],
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'vect__max_df': np.arange(0.1, 1.1, 0.1),
        'vect__min_df': np.arange(0, 6, 1),
        'tfidf__use_idf': (True, False),
        # 'tfidf__sublinear_tf': (True, False),
        # 'tfidf__smooth_idf': (True, False),
        # 'tfidf__norm': ('l1', 'l2'),
        'clf__alpha': (1e-2, 1e-3),
        'clf__binarize': (None, 0.0, 0.5, 1.0),
    },
    {
        # ALL PARAMS READY TO RUN 3
        'clf': [SGDClassifier(penalty='l2', max_iter=1000, tol=1e-3)],
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'vect__max_df': np.arange(0.1, 1.1, 0.1),
        'vect__min_df': np.arange(0, 6, 1),
        'tfidf__use_idf': (True, False),
        # 'tfidf__sublinear_tf': (True, False),
        # 'tfidf__smooth_idf': (True, False),
        # 'tfidf__norm': ('l1', 'l2'),
        'clf__loss': ('hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'),
        'clf__alpha': (1e-3, 1e-4, 1e-5),
        # 'clf__penalty': ('l1', 'l2'),
        # 'clf__l1_ratio': np.arange(0.01, 0.5, 0.05),
        'clf__fit_intercept': (True, False),
        # 'clf__max_iter': (500, 1000),
        # 'clf__tol': (1e-2, 1e-3),
        'clf__shuffle': (True, False),
        # 'clf__eta0': np.arange(0.0, 1.1, 0.1),
        'clf__power_t': np.arange(0.0, 1.1, 0.1),
        'clf__early_stopping': (True, False),
        'clf__validation_fraction': np.arange(0.1, 1.1, 0.1),
        'clf__n_iter_no_change': np.arange(2, 8, 1),
        'clf__average': (True, False)
    },
    {
        # ALL PARAMS READY TO RUN 4
        'clf': [SVC(gamma='auto', tol=1e-3, cache_size=700, random_state=42)],
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'vect__max_df': np.arange(0.1, 1.1, 0.1),
        'vect__min_df': np.arange(0, 6, 1),
        'tfidf__use_idf': (True, False),
        # 'tfidf__sublinear_tf': (True, False),
        # 'tfidf__smooth_idf': (True, False),
        # 'tfidf__norm': ('l1', 'l2'),
        'clf__kernel': ('linear', 'rbf', 'precomputed'),
        'clf__shrinking': (True, False)
    },
    {
        # ALL PARAMS READY TO RUN 5
        # ONLY KERNEL=POLY
        'clf': [SVC(gamma='auto', tol=1e-3, cache_size=700, random_state=42, kernel='poly')],
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'vect__max_df': np.arange(0.1, 1.1, 0.1),
        'vect__min_df': np.arange(0, 6, 1),
        'tfidf__use_idf': (True, False),
        # 'tfidf__sublinear_tf': (True, False),
        # 'tfidf__smooth_idf': (True, False),
        # 'tfidf__norm': ('l1', 'l2'),
        'clf__degree': np.arange(1, 6, 1),
        'clf__coef0': np.arange(0.0, 1.1, 0.1),
        'clf__shrinking': (True, False)
    },
    {
        # ALL PARAMS READY TO RUN 6
        # ONLY KERNEL=SIGMOID
        'clf': [SVC(gamma='auto', tol=1e-3, cache_size=700, random_state=42, kernel='sigmoid')],
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'vect__max_df': np.arange(0.1, 1.1, 0.1),
        'vect__min_df': np.arange(0, 6, 1),
        'tfidf__use_idf': (True, False),
        # 'tfidf__sublinear_tf': (True, False),
        # 'tfidf__smooth_idf': (True, False),
        # 'tfidf__norm': ('l1', 'l2'),
        'clf__coef0': np.arange(0.0, 1.1, 0.1),
        'clf__shrinking': (True, False)
    },
    {
        # ALL PARAMS READY TO RUN 7
        'clf': [KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300,
                       tol=1e-4, precompute_distances='auto', copy_x=True)],
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'vect__max_df': np.arange(0.1, 1.1, 0.1),
        'vect__min_df': np.arange(0, 6, 1),
        'tfidf__use_idf': (True, False),
        # 'tfidf__sublinear_tf': (True, False),
        # 'tfidf__smooth_idf': (True, False),
        # 'tfidf__norm': ('l1', 'l2'),
        # 'clf__n_init': (10, 15),
        # 'clf__max_iter': (300),
        # 'clf__tol': (1e-4),
        # 'clf__precompute_distances': ('auto', True, False),
        # 'clf__copy_x': (True, False),      # with precompute_distances=True
    },
    {
        # ALL PARAMS READY TO RUN 8
        'clf': [DecisionTreeClassifier(presort=False)],
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'vect__max_df': np.arange(0.1, 1.1, 0.1),
        'vect__min_df': np.arange(0, 6, 1),
        'tfidf__use_idf': (True, False),
        # 'tfidf__sublinear_tf': (True, False),
        # 'tfidf__smooth_idf': (True, False),
        # 'tfidf__norm': ('l1', 'l2'),
        'clf__criterion': ('gini', 'entropy'),
        'clf__splitter': ('best', 'random'),
        # 'clf__max_depth': (None, '%INTEGER'),
        'clf__min_samples_split': np.arange(2, 7, 1),
        'clf__min_samples_leaf': np.arange(1, 6, 1),
        # 'clf__min_weight_fraction_leaf': (0.),
        'clf__max_features': ('auto', 'log2', None, np.arange(0.1, 1.1, 0.1)),
        # 'clf__random_state': (None),
        # 'clf__max_leaf_nodes': (None),
        # 'clf__min_impurity_decrease': (0.),
        'clf__class_weight': (None, 'balanced'),
        # 'clf__presort': (True, False)
    }
]
scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score), 'F1-Score': make_scorer(f1_score)}
gs_clf = GridSearchCV(text_clf, parameters[0], cv=5, iid=False, n_jobs=-1, verbose=51, error_score=0,
                      scoring=scoring, refit='AUC', return_train_score=True)

gs_clf = gs_clf.fit(X_train, y_train)

# test
predict = gs_clf.predict(X_test)
numpy_mean = np.mean(predict == y_test)
print(gs_clf.best_estimator_._final_estimator,  numpy_mean)
classification_report = metrics.classification_report(y_test, predict)
print(classification_report)
confusion_matrix = metrics.confusion_matrix(y_test, predict)
t = Texttable()
t.add_rows([["", "Is not Sarcastic (PREDICTED)", "Is Sarcastic (PREDICTED)", "TOTAL"],
            ["Is not Sarcastic (REAL)", confusion_matrix[0][0], confusion_matrix[0][1], confusion_matrix[0][0] + confusion_matrix[0][1]],
              ["Is Sarcastic (REAL)", confusion_matrix[1][0], confusion_matrix[1][1], confusion_matrix[1][0] + confusion_matrix[1][1]],
              ["TOTAL", confusion_matrix[0][0] + confusion_matrix[1][0], confusion_matrix[0][1] + confusion_matrix[1][1],
              confusion_matrix[0][0] + confusion_matrix[1][0] + confusion_matrix[0][1] + confusion_matrix[1][1]]])
print(t.draw())

print(gs_clf.best_estimator_)

# Salvataggio su file dei risultati come da output console

file_name = re.search("^[\word]*", gs_clf.best_estimator_.steps[2][1].__str__()).group()
if file_name == 'SVC':
    file_result = open('../results/'+file_name+'_'+gs_clf.best_estimator_.steps[2][1].__getattribute__('kernel').__str__()+'.txt', 'w')
else:
    file_result = open('../results/'+file_name+'.txt', 'w')


file_result.writelines([gs_clf.param_grid.__str__() + '\n', gs_clf.best_estimator_._final_estimator.__str__() + ' ',
                        'NumPy Mean: ' + numpy_mean.__str__() + '\n', classification_report + '\n',
                        t.draw() + '\n', gs_clf.best_estimator_.__str__() + '\n'])

file_result.close()

# Scrittura su file del dell'attributo cv_results_ di gs_clf (GridSearch)
if file_name == 'SVC':
    file_cv_results = open('../results/'+file_name+'_'+gs_clf.best_estimator_.steps[2][1].__getattribute__('kernel').__str__()+'_cv_results.pck', 'wb')
else:
    file_cv_results = open('../results/'+file_name+'_cv_results.pck', 'wb')

pickle.dump(gs_clf.cv_results_, file_cv_results)

# Per il recupero dell'intera struttura cv_results_ generata dal GridSearch
# file_cv_results = open('../results/NOMEFILE_cv_results.pck', 'rb')
# prova_lettura = pickle.load(file_cv_results)


plt.figure(figsize=(30, 20))
plt.title("GridSearchCV evaluating using multiple scorers simultaneously", fontsize=16)
plt.xlabel("Tests")
plt.ylabel("Score")
ax = plt.gca()
ax.set_ylim(0.75, 1)
plt.grid(False)
plt.plot(gs_clf.cv_results_.get('mean_test_AUC'))
plt.plot(gs_clf.cv_results_.get('mean_test_Accuracy'))
plt.plot(gs_clf.cv_results_.get('mean_test_F1-Score'))

X_axis = np.arange(0, gs_clf.cv_results_.get('mean_test_AUC').__len__(), 1)

for scorer, color in zip(sorted(scoring), ['g', 'k', 'b']):
    best_index = np.nonzero(gs_clf.cv_results_['rank_test_%s' % scorer] == 1)[0][0]
    best_score = gs_clf.cv_results_['mean_test_%s' % scorer][best_index]

    # Plot di una linea tratteggiata sull'ascissa del best score
    ax.plot([X_axis[best_index], ] * 2, [0, best_score],
            linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

    # Scrittura del valore migliore
    ax.annotate("%0.2f" % best_score,
                (X_axis[best_index], best_score))
plt.legend(['AUC', 'Accuracy', 'F1-Score'])
plt.savefig('../results/'+file_name+'.png')
if file_name == 'SVC':
    plt.savefig('../results/'+file_name+'_'+gs_clf.best_estimator_.steps[2][1].__getattribute__('kernel').__str__()+'.png')
else:
    plt.savefig('../results/'+file_name+'.png')
plt.show()
plt.close()
