#!/usr/bin/env python

import sys
import numpy as np
from sklearn import cross_validation, metrics

instr = sys.argv[1]


### Data Matrix Preprocessing

# Import data file as a list.
data = open('train.tsv', 'r').read().split('\n')[ : -1]

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
    


### Set model assignments

# Do we perform SVD (only for random forest model, boolean)?
svd_bool = False

## Random Forest Model
if instr == 'rf':

    # Check svd, and set number of desired svd components. We set an arbitrary 500,
    # however, this quantity can be trained using the same cross-validation process
    # outlined below.
    svd_bool = True
    svd_comp = 500

    # Import random forest classifier.
    from sklearn.ensemble import RandomForestClassifier

    # Set number of trees for ensemble (passed in as an optional shell parameter, 
    # default = 100).
    if len(sys.argv) > 2:
        n_tree = sys.argv[2]
    else:
        n_tree = 100

    # Initialize random forest classifier, and allow parallel processing.
    model = RandomForestClassifier(n_estimators = n_tree, n_jobs = -1)


## Multinomial Naive Bayes Classifier
 
elif instr == 'nb':

    # Import Naive Bayes Classifier

    from sklearn.naive_bayes import MultinomialNB
    
    # Set additive smoothing parameter (passed in as an optional shell parameter,
    # default = 1.0).
    if len(sys.argv) > 2:
        alpha = sys.argv[2]
    else:
        alpha = 1.0

    # Initialize Multinomial Naive Bayes classifier.
    model = MultinomialNB(alpha = alpha)

    

### Cross Validation MSE estimation

# Set number of desired folds.
k = 7

# Get set of k-folds.
k_folds = cross_validation.StratifiedKFold(response, k, shuffle = True)

# Initialize three different loss functions. We will initialize a Mean Squared Error
# loss function as this is what is required for the evaluation engine. However, since
# we are treating this as a classification problem we will also consider the accuracy
# score loss function.
MSE_loss = metrics.mean_squared_error
accuracy_loss = metrics.accuracy_score

metric_estimates = []
iter_count = 1

# Begin k_fold estimation of the above metrics (given the set parameters).


for train_id, test_id in k_folds:
    
    # Keep track of fold number through computation
    print(str(iter_count))
    iter_count += 1

    # Separate fold training and test sets.
    train_data, test_data = data[train_id], data[test_id]
    train_response, test_response = response[train_id], list(response[test_id])

    # Perform an SVD dimensionality reduction on our word count matrix if required.
    if svd_bool:
        from sklearn.decomposition import TruncatedSVD
        train_data = TruncatedSVD(n_components = svd_comp).fit_transform(train_data)
        test_data = TruncatedSVD(n_components = svd_comp).fit_transform(test_data)
        
    #  Fit model.
    model = model.fit(train_data, train_response)
    
    # Obtain set of predicted values.
    pred_response = list(model.predict(test_data))

    # Append set of fold predicted MSE values to metric_estimates.
    metric_estimates.append([MSE_loss(test_response, pred_response), 
                             accuracy_loss(test_response, pred_response)])

# Finally, we will print out set of predicted responses, as well as average estimate over
# set of k-folds.
MSE_estimate, accuracy_score_estimate = 0, 0

print('Fold\tMSE\tAccuracy')
for i in range(k):
    MSE_estimate += metric_estimates[i][0]
    accuracy_score_estimate += metric_estimates[i][1]
    print(str(i + 1) + '\t' + str(metric_estimates[i][0]) + '\t' + 
          str(metric_estimates[i][1]))
    
print()
print('MSE Estimate\tAccuracy Score Estimate')
print(str(MSE_estimate / k) + '\t' + str(accuracy_score_estimate / k))


    





