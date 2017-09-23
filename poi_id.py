#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

# Count the NaN values for each people and total number of POI
list_ppl_NaN = []
NbrPOI = 0
for name in data_dict:
    Nancount = 0
    for feat in data_dict[name]:
        if (data_dict[name][feat] == 'NaN'):
            Nancount += 1
    list_ppl_NaN.append([name, Nancount])
    if (data_dict[name]['poi'] == True):
        NbrPOI += 1
from operator import itemgetter

list_ppl_NaN.sort(key=itemgetter(1),reverse=True)

print "Number of POI",NbrPOI


# Plot the distribution of NaN
list_histo_Nan = []
for el in list_ppl_NaN:
    list_histo_Nan.append(el[1])
import matplotlib.pyplot as plt

plt.hist(list_histo_Nan,bins=20,range=(2,20),align='mid')
plt.xlabel('Nbr of NaN values')
plt.ylabel('Nbr of keys ')
plt.title('Distribution of the number of NaN values for each person')
plt.show()



# Remove people with more than 16 NaN values
count_remove = 0

for el in list_ppl_NaN:
    if (el[1]> 16):
        print el[0]
        print el[1]
        print data_dict[el[0]]['poi']
        data_dict.pop( el[0], 0 )
        count_remove +=1
print count_remove, "keys have been removed"


# Remove the specific TOTAL key
data_dict.pop( 'TOTAL', 0 )




### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
     
for name in my_dataset:
  
    # Ratio of emails that is from POI
    if (my_dataset[name]['from_this_person_to_poi']=='NaN') or \
    (my_dataset[name]['from_messages']=='NaN'):
      my_dataset[name]['ratio_from_poi'] = 0
    else:  
      my_dataset[name]['ratio_from_poi'] = (1.0*my_dataset[name]
      ['from_this_person_to_poi']/my_dataset[name]['from_messages'])   
        
    # Ratio of emails that is to POI  
    if (my_dataset[name]['from_poi_to_this_person']=='NaN') or \
    (my_dataset[name]['to_messages']=='NaN'):
      my_dataset[name]['ratio_to_poi'] = 0
    else:  
      my_dataset[name]['ratio_to_poi'] = (1.0*my_dataset[name]
      ['from_poi_to_this_person']/my_dataset[name]['to_messages'])   


# I haved removed by instinct email address and other
temp_features_list = ['poi','salary','deferral_payments', 'total_payments',
                 'loan_advances', 'bonus', 'restricted_stock_deferred',
                 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'long_term_incentive',
                 'restricted_stock', 'director_fees','to_messages',
                 'from_poi_to_this_person', 'from_messages',
                 'from_this_person_to_poi', 'shared_receipt_with_poi','ratio_from_poi',
                 'ratio_to_poi']

data = featureFormat(my_dataset, temp_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Scale features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
features = scaler.fit_transform(features)



# Select K best
from sklearn.feature_selection import SelectKBest, f_classif

Kfeat = 5
k_best = SelectKBest(f_classif,k=Kfeat)
k_best.fit(features, labels)

results_list = zip(temp_features_list[1:], k_best.scores_)
results_list = sorted(results_list, key=lambda x: x[1], reverse=True)
print "K-best features:", results_list[0:Kfeat]




# Plot the features
list_lab_features = []
list_score_features = []
for el in results_list:
    list_lab_features.append(el[0])
    list_score_features.append(el[1])

import numpy as np

N=len(list_score_features)
ind = np.arange(N)
width = 0.5       

plt.bar(ind, list_score_features,   width, color='b')

plt.ylabel('Scores')
plt.title('K Best feature scores')
plt.xticks(ind+width/2., list_lab_features,rotation='vertical')
plt.yticks(np.arange(0,25,2))
plt.show()



# get the features
for f in results_list[0:Kfeat]:
    features_list.append(f[0])

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)




### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

"""
### Kmeans
from sklearn.cluster import KMeans
clf = KMeans(n_clusters=2)
"""


### Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()


"""
### Decision Tree
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
"""

"""
### Random Forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state = 42)
"""


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

def RandomForestclfTuned():
    
    # tuning RF algorithm
    from sklearn import grid_search
    from sklearn.ensemble import RandomForestClassifier
    
    
    # parmeters for random forest
    
    RFparameters = {'random_state':[42],'n_estimators':[10,20,30,40,50,100],
                    'max_features':[0.5,1.0,"sqrt","log2"],
                    'min_samples_split':[1,2,3,5,8,10], "min_samples_leaf":
                        [1,2,3,5,8,10]}
    
    RFclf = RandomForestClassifier()
    clf = grid_search.GridSearchCV(RFclf, RFparameters)
    
    from sklearn.cross_validation import train_test_split
    features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


    clf.fit(features_train, labels_train)
    
    return(clf.best_estimator_)



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
