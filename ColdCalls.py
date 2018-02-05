
# coding: utf-8

# In[74]:

# Call needed libraries 
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_val_predict
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model


from sklearn.neighbors  import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
   
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    


    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
class_names = ['Success','Failure']

if __name__ == '__main__':

    # Import data to pandas
    TrainX = pd.read_csv('train_mod_X_Python.csv')
    TrainY = pd.read_csv('train_mod_y_Python.csv')

    # Drop the irrelevant duplicate column with Id info
    TrainX = TrainX.drop(['Unnamed: 0'], axis = 1)
    TrainY = TrainY.drop(['Unnamed: 0'], axis = 1)

    # Check for missing values
    # (Already taken care of in R)
    TrainX.isnull().sum()

    # Convert the factor setting
    # of R to dummy (binary) columns
    Job = pd.get_dummies(data = TrainX['Job'], prefix = "Job")

    Marital= pd.get_dummies(data = TrainX['Marital'], prefix = "Marital")

    Education= pd.get_dummies(data = TrainX['Education'], prefix="Education")

    Communication = pd.get_dummies(data = TrainX['Communication'], prefix = "Communication")

    LastContactMonth = pd.get_dummies(data = TrainX['LastContactMonth'], prefix= "LastContactMonth")

    Outcome = pd.get_dummies(data = TrainX['Outcome'], prefix = "Outcome")

    CallHour = pd.get_dummies(data = TrainX['CallHour'], prefix = "CallHour")

    AgeBucket = pd.get_dummies(data = TrainX['AgeBucket'], prefix = "AgeBucket")

    BalanceBucket = pd.get_dummies(data = TrainX['BalanceBucket'], prefix = "BalanceBucket")


    # Drop original level columns
    TrainX_dropped = TrainX.drop(['Job',
                                'Marital',
                                'Education',
                                'Communication',
                                'LastContactMonth',
                                'Outcome', 
                                'CallHour',
                                'AgeBucket',
                                'BalanceBucket'],
                                axis=1)

    #Concatenate the dropped dataframe with dummy columns
    TrainX = pd.concat([TrainX_dropped, 
                        Job, 
                        Marital, 
                        Education, 
                        Communication, 
                        LastContactMonth, 
                        Outcome, 
                        CallHour, 
                        AgeBucket, 
                        BalanceBucket],
                    axis=1)

    # Save and drop the customer ID array for further use
    ID = TrainX['Id'].values
    TrainX = TrainX.drop(['Id'],axis=1)

    X_train, X_test, y_train, y_test = train_test_split(TrainX, TrainY, test_size=0.20,random_state=42)

    # Create corresponding numpy matrices (NOT NEEDED??)
    mTrainX = X_train.as_matrix(columns=None)
    mTrainY = y_train.as_matrix(columns=None)
    y_train_array = np.reshape(mTrainY, 3184)

    # Create corresponding numpy matrices (NOT NEEDED??)
    mTestX = X_test.as_matrix(columns=None)
    mTestY = y_test.as_matrix(columns=None)
    y_test_array = np.reshape(mTestY, 796)

    # Preprocess data to zero-mean & unit-variance
    #from sklearn import preprocessing
    #X_train = preprocessing.scale(mTrainX)
    #X_test = preprocessing.scale(mTestX)

    #Splitting the Training and Testing data having 20% of Test data

    # LR_1_CVS = cross_val_score(estimator=linear_model.LogisticRegression(), X=mTrainX, y=y_train_array, cv=5)
    # score_knn = cross_val_score(KNeighborsClassifier(n_neighbors = 6), X=mTrainX, y=y_train_array, cv=5)
    # score_svm = cross_val_score(svm.SVC(), X=mTrainX, y=y_train_array, cv=5)
    # score_rfc = cross_val_score(RandomForestClassifier(n_estimators=1000, max_depth=None, min_samples_split=10,class_weight="balanced"), X=mTrainX, y=y_train_array, cv=5)

    # print('LR_1_CVS:', LR_1_CVS)
    # print('score_knn:', score_knn)
    # print('score_svm:', score_svm)
    # print('score_rfc:', score_rfc)


    # Drawing confusion matrixes
    # LogRed
    LR = linear_model.LogisticRegression()
    LR.fit(mTrainX,y_train_array)
    y_pred = LR.predict(mTestX)
    print(classification_report(y_test_array, y_pred))

    cm = confusion_matrix(y_test_array, y_pred)
    plot_confusion_matrix(cm, classes=class_names, title='LogReg - Confusion matrix')

    ##########
    # Random Forest Classifier
    rfc = RandomForestClassifier(n_estimators=1000, max_depth=None, min_samples_split=10,class_weight="balanced")
    rfc.fit(mTrainX,y_train_array)
    y_pred = rfc.predict(mTestX)
    print(classification_report(y_test_array, y_pred))

    cm = confusion_matrix(y_test_array, y_pred)
    plot_confusion_matrix(cm, classes=class_names, title='Random Forest Classifier - Confusion matrix')
