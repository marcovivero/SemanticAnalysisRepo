#!/usr/bin/env python

import sys
import numpy as np
from sklearn import cross_validation, metrics


### Data Matrix Preprocessing

# Import data file as a list.
data = open('train2.tsv', 'r').read().split('\n')[ : -1]

# Function used to obtain response vector from data_list.
def get_response(data = data):
    return np.array([int(x.split('\t')[3]) for x in data])

# Function used to obtain word count data matrix from data_list.
def get_data(data = data, max_df = 0.90, min_df = 5):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(max_df= max_df, min_df = min_df)
    data = [x.split('\t')[2] for x in data]
    return vectorizer.fit_transform(data)

# Obtain response vector and data matrix.
response = get_response()
data = get_data()




### Cross Validation MSE estimation

def cross_validate(model_type, data, response, folds, metrics,
                   alpha = '', n_tree= ''):
    metric_estimates = []
    for train_id, test_id in folds:

        # Separate fold training and test sets.
        train_data, test_data = data[train_id], data[test_id]
        train_response, test_response = response[train_id], list(response[test_id])
        
        ### Set model assignments
        ## Random Forest Model
        if model_type == 'rf':

            # Import random forest classifier.
            from sklearn.ensemble import RandomForestClassifier

            # Initialize random forest classifier, and allow parallel processing.
            model = RandomForestClassifier(n_estimators = n_tree, n_jobs = -1)

            ## Multinomial Naive Bayes Classifier
 
        elif model_type == 'nb':

            # Import Naive Bayes Classifier

            from sklearn.naive_bayes import MultinomialNB

            # Initialize Multinomial Naive Bayes classifier.
            model = MultinomialNB(alpha = alpha)

        #  Fit model.
        model = model.fit(train_data, train_response)    

        # Obtain set of predicted values.
        pred_response = list(model.predict(test_data))

        # Append set of fold predicted MSE values to metric_estimates.
        metric_estimates.append([f(test_response, pred_response) for f in metrics])
        
    return np.mean(np.array(metric_estimates), axis = 0)


###  Automated Parameter Tuning
### Follows the algorithm given with accompanying PDF.

def tuneNaiveBayes(data, response, alpha_0, size = 100, 
                   burnin = 0, n_folds = 7, 
                   prior = lambda : np.random.uniform(0, 1000)):
    alpha_old = alpha_0
    folds = cross_validation.StratifiedKFold(response, n_folds, shuffle = True)
    MSE_old = cross_validate('nb', data, response, folds,
                             [metrics.mean_squared_error], alpha = alpha_old)
    n_accept, n_total, samps = 0, 0, []
    
    while n_accept < burnin + size:
        alpha_new = prior()
        MSE_new = cross_validate('nb', data, response, folds,
                                 [metrics.mean_squared_error], alpha = alpha_new)
        if min(1, (1 / MSE_new) / (1 / MSE_old)) > np.random.uniform():
            n_accept += 1
            alpha_old, MSE_old = alpha_new, MSE_new
            if n_accept > burnin:
                samps.append([alpha_old, MSE_old])
        n_total += 1
    return samps
        
        


