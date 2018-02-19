from draw import draw_confusion_matrix, draw_correlation_heatmap, draw_roc, draw_rfecv, draw_feature_importance, plot_roc_values
from data_preprocess import feature_reduction
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data Processing:
from sklearn.preprocessing import StandardScaler, scale
from sklearn.feature_selection import RFECV

from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_val_predict

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV


names = [
    "Nearest Neighbors", # 0.63
    #"Decision Tree", # 0.54
    "Neural Net", # 0.61
    "AdaBoost", # 0.64
    'LDA', # 0.62
    'LogReg', # 0.68
    'XGBoost', # 0.59
    #"QDA", # 0.55
    #"Naive Bayes", # 0.52 
    #"Random Forest" # 0.53
    ]
tuned_parameters = [
    {'n_neighbors': [3, 9, 14, 17, 18, 19, 25], 'p' : [1, 2]},
    #{'max_depth': [5, 10, 20]},
    {'alpha': [0.01, 0.1, 1,], 'activation' : ['identity', 'logistic', 'tanh', 'relu'], 'hidden_layer_sizes': [100, 200, 300]},
    {'n_estimators': [10, 30, 50, 100, 200], 'learning_rate': [0.01, 0.1, 1, 10, 100]},
    {'n_components': [2, 6, 12, 30]},
    {'C': [0.01, 0.1, 1, 10, 100, 1000, 3000], 'penalty': ['l1', 'l2']},
    #{'n_estimators': [3, 10, 100, 300], 'learning_rate': [0.001, 0.01, 0.1, 1, 10]},
    #{'tol': [1.0e-3, 1.0e-4, 1.0e-5]}, # QDA
    #{}, # Naive Bayes
    #{'n_estimators': [3, 10, 30, 90], 'max_features': ['auto', 'log2', None]} # Random Forest
    ]
classifiers_default = [
    KNeighborsClassifier(), #{n_neighbors = 18, p = 1}
    #DecisionTreeClassifier(), #max_depth=10
    MLPClassifier(), #{activation = 'tanh', alpha=0.01, hidden_layer_sizes = 200}
    AdaBoostClassifier(), #{learning_rate = 0.1, n_estimators = 200}
    LinearDiscriminantAnalysis(), #{n_components=2}
    LogisticRegression(), #{C = 1000, penalty = 'l2'}
    #GradientBoostingClassifier(), #{n_estimators=300,learning_rate=0.1}
    #QuadraticDiscriminantAnalysis(),
    #GaussianNB(),
    #RandomForestClassifier() #
    ] 

classifiers_third = [
    KNeighborsClassifier(n_neighbors=25, p=1), # n_neighbors=25, p=1
    #DecisionTreeClassifier(), #max_depth=5
    MLPClassifier(activation='logistic', alpha=0.1, hidden_layer_sizes=300), # activation='logistic', alpha=0.1, hidden_layer_sizes=300
    AdaBoostClassifier(learning_rate = 1, n_estimators = 30), # learning_rate = 1, n_estimators = 30
    LinearDiscriminantAnalysis(n_components=2), # n_components=2
    LogisticRegression(C = 0.01, penalty = 'l2'), # C = 0.01, penalty = 'l2'
    #GradientBoostingClassifier(n_estimators=300,learning_rate=0.01), #{n_estimators=300,learning_rate=0.01}
    #QuadraticDiscriminantAnalysis(), # 'tol': 0.001
    #GaussianNB(),
    #RandomForestClassifier() # 'max_features': 'auto', 'n_estimators': 90
    ] 

def default_classifiers(X_train, X_test, y_train, y_test, data_name):
    cross_valid = 10
    #no_rfecv = ['Nearest Neighbors', 'RBF SVM', 'Neural Net', 'Naive Bayes', 'QDA']

    for clf, name in zip(classifiers_default, names):
        print('######## {} - with CV #########'.format(name))
        clf.fit(X_train, y_train)

        score = clf.score(X_test, y_test)
        print(score,'Test')
        cv_scores = cross_val_score(clf, X_train, y_train, cv=cross_valid)
        print(np.mean(cv_scores), 'CV')

        # Plot ROC values:
        plt.figure(2)
        plot_roc_values(clf, X_test, y_test, name)

        # Print Classification Report:
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))

        # Draw Confusion Matrix:
        # cm = confusion_matrix(y_test, y_pred)
        # plt.figure(3)
        # draw_confusion_matrix(cm, title='{} | {} - Confusion matrix'.format(data_name, name))
    
    plt.figure(2)
    draw_roc(data_name)
    
    return 

def grid_search_classifiers(X_train, X_test, y_train, y_test, data_name):
    optimal_parameters =[]
    cross_validation = 10
    scores = [
        #'recall',
        'precision'
        ]
    for clf_orig, name, tuned_parameter in zip(classifiers_default, names, tuned_parameters):
        if name is '':
            return
        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print('##########   {}   ##########'.format(name))

            clf = GridSearchCV(clf_orig, tuned_parameter, cv=cross_validation,
                            scoring='%s_macro' % score)
            clf.fit(X_train, y_train)
            optimal_parameters.append(clf.best_params_)

            print("Best parameters set found on development set:")
            print(clf.best_params_)
            print()
            y_true, y_pred = y_test, clf.predict(X_test)
            print(classification_report(y_true, y_pred))
            print()
    return optimal_parameters

def print_voting(X, y, X_test, y_test, data_name, parameters=None):
    print('###### Voting - {} ######'.format(data_name))
    # Voting Ensemble for Classification
    from sklearn.ensemble import VotingClassifier
    kfold = KFold(n_splits=10, random_state=42)
    # create the sub models
    estimators = []
    for name, clf in zip(names, classifiers_third):
        estimators.append((name, clf))    
    # create the ensemble model
    ensemble = VotingClassifier(estimators, voting='soft')
    results = cross_val_score(ensemble, X, y, cv=kfold)
    ensemble.fit(X, y)
    y_true, y_pred = y_test, ensemble.predict(X_test)
    print("Detailed classification report:")
    print()
    print(classification_report(y_true, y_pred))
    print('CV-result', results.mean())
    print()
    print('-----------------')
    print()

    # Draw Voting Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    draw_confusion_matrix(cm, title='{} | {} - Confusion matrix'.format(data_name, 'Voting'))


if __name__ == '__main__':

    # Import data to pandas
    df = pd.read_csv('finaldata.csv')

    #####
    # Shaping Data
    first_drop = ['Id', 'Age', 'AgeBinEW', 'Balance', 'BalanceBin', 'LastContactDay', 'WithinMonth', 'PrevAttempsToDaysPassed', 'Id_duplicate', 'PAtDPEF'] # Waste
    second_drop = ['Communication', 'LastContactMonth', "NoOfContacts", "CallDuration", "CallHour", 'WithinMonthEF']
    third_drop = ["PrevAttempts", "Outcome", "DaysPassed"]

    df_first_data = df.drop(first_drop, axis=1)
    df_second_data = df_first_data.drop(second_drop, axis=1)
    df_third_data = df_second_data.drop(third_drop, axis=1)

    datas = [df_first_data, df_second_data, df_third_data]
    data_names = ['First Data', 'Second Data', 'Third Data']
    #####

    i = 0
    for data, data_name in zip(datas, data_names):
        if i < 2:
            i+=1
            continue
        print('\n\n>>>>>>>>>>>>>>>> {} <<<<<<<<<<<<<<<<<\n'.format(data_name))
        data = pd.get_dummies(data, drop_first=True)
        X = data.drop(['CarInsurance'], axis=1)
        y = data.CarInsurance

        # Data preprocessing:
        #voting(X, y)
        #feature_reduction(X, 0.8)
        #draw_correlation_heatmap(X, data_name)
        data = StandardScaler().fit_transform(data)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=None)

        print_voting(X, y, X_test, y_test, data_name)
        #draw_feature_importance(X_train, y_train, data_name)
        # Drawing confusion matrixes
        #optimal_parameters = grid_search_classifiers(X_train, X_test, y_train, y_test, data_name)
        #print('optimal_parameters:', optimal_parameters)
        #default_classifiers(X_train, X_test, y_train, y_test, data_name)
