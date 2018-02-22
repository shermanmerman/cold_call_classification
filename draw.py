import numpy as np
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
import pandas as pd

# draw_feature_importance:
from sklearn.ensemble import ExtraTreesClassifier
# plot_roc_values:
from sklearn.metrics import roc_curve, roc_auc_score

def draw_confusion_matrix(cm,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    classes = ['Success','Failure']
   
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
    plt.savefig('figures/{}.pdf'.format(title), bbox_inches='tight')
    #plt.show()
    plt.clf()


def draw_correlation_heatmap(X, data_name):
    corrMatrix = X.corr()
    sns.heatmap(corrMatrix, mask=np.zeros_like(corrMatrix, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True, xticklabels=False, yticklabels=False)
    plt.title("Correlation_Heatmap-{}".format(data_name))
    plt.savefig('figures/Correlation_Heatmap_{}.pdf'.format(data_name), bbox_inches='tight')
    #plt.show()
    plt.clf()

def draw_roc(data_name):
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Graph')
    plt.legend(loc="lower right")
    plt.savefig('figures/ROC_Graph-{}.pdf'.format(data_name), bbox_inches='tight')
    #plt.show()
    plt.clf()

def draw_rfecv(data_name):
    plt.title('Recursive feature elimination')
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.legend(loc="lower right")
    plt.savefig('figures/Recursive_feature_elimination-{}.pdf'.format(data_name), bbox_inches='tight')
    #plt.show()
    plt.clf()

def draw_feature_importance(X_train, y_train, data_name):
    # Using ExtraTreesClassifier model function
    model = ExtraTreesClassifier()
    model.fit(X_train, y_train)
    # Printing important features in the model
    #print(model.feature_importances_)
    importances = model.feature_importances_
    feat_names = X_train.columns

    # Displaying the feature importances as a chart by sorting it in the order of importances
    indices = np.argsort(importances)[::-1]
    plt.figure(4)
    plt.figure(figsize=(12,6))
    plt.title("Feature importances - {}".format(data_name))
    plt.bar(range(len(indices)), importances[indices], color='lightblue',  align="center")
    plt.step(range(len(indices)), np.cumsum(importances[indices]), where='mid', label='Cumulative')
    plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical',fontsize=14)
    plt.xlim([-1, len(indices)])
    plt.savefig('figures/Feature_importance-{}.pdf'.format(data_name), bbox_inches='tight')
    #plt.show()
    plt.clf()

def plot_roc_values(clf, X_test, y_test, name):
    y_scores = clf.predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_score = roc_auc_score(y_test, y_scores)
    plt.plot(fpr, tpr, label='{:.2f} - {}'.format(roc_score, name))