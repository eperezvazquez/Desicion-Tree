#pip install scikit-learn --upgrade
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import plotly.express as px
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
import plotly.offline as py
import plotly.io as pio
import plotly.graph_objs as go
import math
from scipy.stats import norm, skew
from utils import Helpers

import warnings 
warnings.filterwarnings('ignore')

#Cargar el data set
df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv')

def set_value (data_value, outcome_value,mean_nodiab,mean_diab):
    if (outcome_value == 0 and data_value==0):
        return mean_nodiab
    elif (outcome_value ==1 and data_value ==0 ):
        return mean_diab
    else:
        return data_value
helpers  = Helpers()
helpers.set_value()
def set_use_mean (name_col,df_raw):
#calc the mean for diabetic and not diabetic that the data is not 0
    meanNoDiab = df_raw[(df_raw[name_col]>0) & (df_raw['Outcome']==0)][name_col].mean()
    meanDiab = df_raw[(df_raw[name_col]>0) & (df_raw['Outcome']==1)][name_col].mean()

    df_raw[name_col] = df_raw.apply(lambda x: set_value(x[name_col], x['Outcome'],meanNoDiab,meanDiab), axis=1)
    return

set_use_mean('Glucose')
set_use_mean('Insulin')
set_use_mean('BMI')
set_use_mean('BloodPressure')
set_use_mean('SkinThickness')
df_raw.describe()

import seaborn as sns
corr = df_raw.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);

# feature selection
feature_cols = ['Pregnancies', 'Insulin', 'BMI', 'Age', 'Glucose', 'BloodPressure', 'DiabetesPedigreeFunction']
x = df_raw[feature_cols]
y = df_raw.Outcome

# split data
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.3, random_state=1)

# build model
classifier = DecisionTreeClassifier()
classifier = classifier.fit(X_train, Y_train)

# predict# confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test, y_pred)
print(confusion_matrix(Y_test, y_pred))

# accuracy
print("Accuracy:", metrics.accuracy_score(Y_test,y_pred))
y_pred = classifier.predict(X_test)
print(y_pred)

##sudo apt install graphviz
##pip install graphviz
#pip install six
#pip install --upgrade scikit-learn==0.20.3
#pip install pydotplus
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
import graphviz
import os

dot_data = StringIO()
export_graphviz(classifier, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())

#Import the required libraries
#https://www.section.io/engineering-education/entropy-information-gain-machine-learning/
from sklearn.datasets import load_iris
from sklearn import tree
from matplotlib import pyplot as plt

# split data Build and fit decision tree 
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.3, random_state=1)

clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4,min_samples_leaf=4)
clf.fit(X_train, Y_train)

#Plot the decision tree.
fig, ax = plt.subplots(figsize=(30, 30))
tree.plot_tree(clf,ax=ax,feature_names=['Pregnancies', 'Insulin', 'BMI', 'Age', 'Glucose', 'BloodPressure', 'DiabetesPedigreeFunction'])
plt.show()

#Step 1 - Import the library - GridSearchCv

from sklearn import decomposition, datasets
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

#Step 2 - Setup the Data
#dataset = datasets.load_wine()
    #X = dataset.data
    #y = dataset.targe

feature_cols = ['Pregnancies', 'Insulin', 'BMI', 'Age', 'Glucose', 'BloodPressure', 'DiabetesPedigreeFunction']
x = df_raw[feature_cols]
y = df_raw.Outcome

#Step 3 - Using StandardScaler and PCA
#StandardScaler is used to remove the outliners and scale the data by making the mean of the data 0 and standard deviation as 1. So we are creating an object std_scl to use standardScaler.
std_slc = StandardScaler()
#We are also using Principal Component Analysis(PCA) which will reduce the dimension of features by creating new features which have most of the varience of the original data.
pca = decomposition.PCA()
#Here, we are using Decision Tree Classifier as a Machine Learning model to use GridSearchCV. So we have created an object dec_tree.
dec_tree = tree.DecisionTreeClassifier()

#Step 5 - Using Pipeline for GridSearchCV
#Pipeline will helps us by passing modules one by one through GridSearchCV for which we want to get the best parameters. So we are making an object pipe to create a pipeline for all the three objects std_scl, pca and dec_tree.
pipe = Pipeline(steps=[('std_slc', std_slc),
                           ('pca', pca),
                           ('dec_tree', dec_tree)])
#Now we have to define the parameters that we want to optimise for these three objects.
#StandardScaler doesnot requires any parameters to be optimised by GridSearchCV.
#Principal Component Analysis requires a parameter 'n_components' to be optimised. 'n_components' signifies the number of components to keep after reducing the dimension.
n_components = list(range(1,x.shape[1]+1,1))
#DecisionTreeClassifier requires two parameters 'criterion' and 'max_depth' to be optimised by GridSearchCV. So we have set these two parameters as a list of values form which GridSearchCV will select the best value of parameter.
criterion = ['gini', 'entropy']
max_depth = [2,4,6,8,10,12]
#Now we are creating a dictionary to set all the parameters options for different objects.
parameters = dict(pca__n_components=n_components,
                      dec_tree__criterion=criterion,
                      dec_tree__max_depth=max_depth)

#Step 6 - Using GridSearchCV and Printing Results
#Before using GridSearchCV, lets have a look on the important parameters.

#estimator: In this we have to pass the models or functions on which we want to use GridSearchCV
#param_grid: Dictionary or list of parameters of models or function in which GridSearchCV have to select the best.
#Scoring: It is used as a evaluating metric for the model performance to decide the best hyperparameters, if not especified then it uses estimator score.
#Making an object clf_GS for GridSearchCV and fitting the dataset i.e X and y
clf_GS = GridSearchCV(pipe, parameters)
clf_GS.fit(x,y)
#Now we are using print statements to print the results. It will give the values of hyperparameters as a result.
print('Best Criterion:', clf_GS.best_estimator_.get_params()['dec_tree__criterion'])
print('Best max_depth:', clf_GS.best_estimator_.get_params()['dec_tree__max_depth'])
print('Best Number Of Components:', clf_GS.best_estimator_.get_params()['pca__n_components'])
print(); print(clf_GS.best_estimator_.get_params()['dec_tree'])




