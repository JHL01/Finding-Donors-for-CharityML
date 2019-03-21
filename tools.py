
# coding: utf-8


#資料整理相關
import numpy as np
import pandas as pd
from time import time

#統計相關
from scipy import stats
from scipy.stats import norm

from sklearn.model_selection import train_test_split,cross_val_score,learning_curve,validation_curve 
from sklearn.metrics import fbeta_score , accuracy_score,roc_auc_score,make_scorer

from IPython.display import display # Allows the use of display() for DataFrames
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV 

def train_predict(learner, features, income): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    # 切分訓練與預測資料
    X_train, X_test, y_train, y_test = train_test_split(features, 
                                                    income, 
                                                    test_size = 0.2, 
                                                    random_state = 0)
               
    print("Training set has {} samples.".format(X_train.shape[0]))
    print("Testing set has {} samples.".format(X_test.shape[0]))

    results = {}

    # Fit the learner to the training data using slicing with 'sample_size' using .fit(training_features[:], training_labels[:])
    start = time() # Get start time
    learner = learner.fit(X_train[:], y_train[:])
    end = time() # Get end time

    # Calculate the training time
    results['train_time'] = round(end-start,4)

    # Get the predictions on the test set(X_test),
    #       then get predictions on the training samples(X_train) using .predict()
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train)
    end = time() # Get end time

    #Calculate the total prediction time
    results['pred_time'] = round(end-start,4)            

#     #Compute F-score on the the first 300 training samples using fbeta_score()
#     results['f_train'] = round(fbeta_score(predictions_train,y_train[:],beta=0.5),3)

#     #Compute F-score on the test set which is y_test
#     results['f_test'] = round(fbeta_score(predictions_test,y_test,beta=0.5),3)

    # Success
    print("{} trained on {} samples".format(learner.__class__.__name__,X_train.shape[0]))

    # Return the results

    return results


scorer = make_scorer(fbeta_score, beta=0.5)
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("fbeta_score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs,scoring=scorer, train_sizes=train_sizes,random_state =0)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Validation score")

    plt.legend(loc="best")
    print("{} test mean scores = {} ".format(estimator.__class__.__name__,test_scores_mean[4]))
    return plt


def features_evaluation(features,income):
    random_state=0
    Grad_n = GradientBoostingClassifier(random_state=random_state)
    Grad_results_n = train_predict(Grad_n, features, income)
    Grad_results_df_n = pd.DataFrame(data=Grad_results_n,index=['GradientBoosting'])
    print(Grad_results_df_n)
    plot_learning_curve(Grad_n, "Learning Curves (GradientBoosting_n)", features, income, ylim=(0.5, 1.01), cv=5, n_jobs=1)

    
def distplot(column):
    sns.distplot(column, fit=norm)
    #skewness and kurtosis
    print("Skewness: %f" % column.skew())
    print("Kurtosis: %f" % column.kurt())

    
def modelfit(alg, features, target, performCV=True, printFeatureImportance=True, cv_folds=5):
    predictors = list(features.columns)
    #Fit the algorithm on the data
    random_state = 0
    alg.fit(features, target)

    #Predict training set:
    dtrain_predictions = alg.predict(features)
    dtrain_predprob = alg.predict_proba(features)[:,1]

    #Perform cross-validation:
    if performCV:
        cv_score = cross_val_score(alg, features, target, cv=cv_folds, scoring=scorer)

    #Print model report:
    print ("\nModel Report")
    print ("F-beta score (Train): %.4g" % fbeta_score(target.values, dtrain_predictions,beta=0.5))
    print ("AUC Score (Train): %f" % roc_auc_score(target, dtrain_predprob))

    if performCV:
        print ("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
              
    #Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)[:10]
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')



    
    