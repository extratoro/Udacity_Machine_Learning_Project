#!/usr/bin/python

import sys
import pickle

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'deferral_payments', 'restricted_stock_deferred', 
                 'deferred_income', 'director_fees', 'exercised_stock_options', 
                 'shared_receipt_with_poi', 'deferred_stock_ratio']

 # You will need to use more features

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


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.


print features_list

#Adding new features
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
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import KernelPCA
from sklearn.feature_selection import SelectKBest

clf = RandomForestClassifier(n_estimators=100, min_samples_split=4, 
                             bootstrap=False, criterion='entropy',
                             class_weight='subsample', max_depth=2)

#clf = Pipeline([('feat', FeatureUnion([('pca', KernelPCA(kernel='rbf', 
#                                                         n_components=2)),
#                                        ('kbest', SelectKBest(k=7))])),
#                ('clf',rand)])


test_classifier(clf, my_dataset, features_list)
print '\n forest:\n'
print clf.feature_importances_


### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)