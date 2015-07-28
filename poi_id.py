#!/usr/bin/python

import sys
import pickle
import pprint


sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 
                 'salary', 
                 'bonus', 
                 'deferred_income', 
                 'expenses', 
                 'exercised_stock_options', 
                 'other', 
                 'long_term_incentive', 
                 'restricted_stock', 
                 'from_poi_to_this_person', 
                 'from_this_person_to_poi',  
                 'shared_receipt_with_poi',
                 'deferral_ratio', 
                 'sent_to_poi_ratio']



### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Task 2: Remove outliers
my_dataset = data_dict

person_null=['TOTAL', 'THE TRAVEL AGENCY IN THE PARK']
for person in data_dict:
    value = 0
    for feat in data_dict[person]:
        if data_dict[person][feat] not in [0, 'NaN', False, '']:
            value = value + 1
    if value == 0:
        person_null.append(person)


for person in person_null:
    my_dataset.pop(person)

print 'number of people in the dataset: '+ str(len(my_dataset.keys()))
print 'Removed data point: ' + str(person_null)
print features_list

### Adding new features
for person in my_dataset:

    #deferal ratio, check for 0 denominator and NaN value
    if my_dataset[person]['deferral_payments'] != 'NaN' and\
    my_dataset[person]['total_payments'] not in [0, 'NaN'] :
        my_dataset[person]['deferral_ratio'] = float(my_dataset[person]['deferral_payments'])/my_dataset[person]['total_payments']
    else:
        my_dataset[person]['deferral_ratio'] = 0
    
    #Exerciced stock ratio, check for 0 denominator and NaN value
    if my_dataset[person]['restricted_stock_deferred'] != 'NaN' and\
    my_dataset[person]['total_stock_value'] not in [0, 'NaN']:
        my_dataset[person]['deferred_stock_ratio'] = float(my_dataset[person]['restricted_stock_deferred'])/my_dataset[person]['total_stock_value']
    else:
        my_dataset[person]['deferred_stock_ratio'] = 0
    
    #email sent to poi ration
    if my_dataset[person]['from_this_person_to_poi'] != 'NaN' and\
    my_dataset[person]['to_messages'] not in [0, 'NaN']:
        my_dataset[person]['sent_to_poi_ratio'] = float(my_dataset[person]['from_this_person_to_poi'])/my_dataset[person]['to_messages']
    else:
        my_dataset[person]['sent_to_poi_ratio'] = 0


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import SelectKBest

### Setting up initial selection, classifier and the pipeline
selection = SelectKBest()
rfc = RandomForestClassifier()
pipeline = Pipeline([('features', selection),
                     ('classifier', rfc)])

### Dict of gridsearch params to go through
parameters = { 'features__k': [5, 7, 10, 'all'],
                'classifier__n_estimators': [50, 100, 200],
               'classifier__min_samples_split': [4, 6, 8], 
                'classifier__criterion': ['entropy', 'gini'],
                'classifier__class_weight': ['subsample', 'auto', None],
                'classifier__max_depth': [2, 4, 6],
                'classifier__warm_start': [False, True]
                }


### Gridsearch and fit to get best params
clf = GridSearchCV(pipeline, parameters, 
                   cv=StratifiedKFold(labels,n_folds=5),
                   scoring='recall')
                   
clf.fit(features, labels)

print '\nbest params\n'
pprint.pprint(clf.best_params_)  
print '\nbest score\n'                     
pprint.pprint(clf.best_score_)
print '\nkbest features\n'
pprint.pprint(clf.best_estimator_.named_steps['features'].scores_)
print '\nrfc features importance\n'
pprint.pprint(clf.best_estimator_.named_steps['classifier'].feature_importances_)


test_classifier(clf.best_estimator_, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf.best_estimator_, my_dataset, features_list)