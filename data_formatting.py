from draw import plot_confusion_matrix
import pandas as pd

from sklearn import linear_model, svm
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_val_predict
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# Heatmap:
import seaborn as sns
sns.set()

dummie_collumns = [ 'Job',
                    'Marital',
                    'Education',
                    'Communication',
                    'LastContactMonth',
                    'Outcome', 
                    'AgeBucket',
                    'BalanceBucket']
duplicates = ['AgeBinEW', 'Balance', 'LastContactDay', 'PrevAttempsToDaysPassed', 'Id_duplicate']


def train_classifier(X_train, X_test, y_train, y_test, classifier):
    # LogRed
    if classifier == 'LogReg':
        clf = linear_model.LogisticRegression()
    # Random Forest Classifier
    elif classifier == 'RandForest':
        clf = RandomForestClassifier(n_estimators=1000, max_depth=None, min_samples_split=10,class_weight="balanced")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    plot_confusion_matrix(cm, title='{} - Confusion matrix'.format(classifier))
    return 



if __name__ == '__main__':

    # Import data to pandas
    df = pd.read_csv('train_python.csv')

    df.drop(duplicates, axis=1, inplace=True)
    df = pd.get_dummies(df, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(df.drop(['CarInsurance'], axis=1), df.CarInsurance, test_size=0.20, stratify=df.CarInsurance)

    # Save and drop the customer ID array for further use
    ID_train = X_train['Id'].values
    df.drop(['Id'], axis=1, inplace=True)

    # calculate the correlation matrix
    corr = X_train.corr()
    # plot the heatmap
    sns.heatmap(corr, 
        xticklabels=X_train.columns,
        yticklabels=X_train.columns)
    plt.show()
    # Drawing confusion matrixes
    train_classifier(X_train, X_test, y_train, y_test, 'LogReg')
    train_classifier(X_train, X_test, y_train, y_test, 'RandForest')
