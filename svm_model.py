import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV

def build_model(X_train, X_test, y_train, y_test):
    print("SVM Model")
    print(X_train.head())
    #print('train shape: ',X_train.shape)
    #print('test shape: ',X_test.shape)

    svm = SVC(gamma='scale', kernel = 'rbf',class_weight='balanced').fit(X_train,y_train)
    print(metrics.confusion_matrix(y_test, svm.predict(X_test)))
    print(metrics.classification_report(y_test, svm.predict(X_test)))
    metrics.plot_confusion_matrix(svm, X_test, y_test)
    plt.savefig("SVM_Confusion_Matrix")


if __name__ == "__main__":
    stock_df = pd.read_csv("sample_stock_data.csv")
    X_train, X_test, y_train, y_test = train_test_split(stock_df.iloc[:,0:-1],stock_df.iloc[:,-1], test_size=0.2)
    X_train1 = X_train.drop(['Unnamed: 0'],axis=1)
    X_test1 = X_test.drop(['Unnamed: 0'],axis=1)

    # Normalized data
    X_train1A = (X_train1-X_train1.mean())/X_train1.std()
    X_test1A = (X_test1-X_train1.mean())/X_train1.std()
    build_model(X_train1A, X_test1A, y_train, y_test)

    # Hyperparameter tuning 
    param_grid = {'C': [0.1, 1, 10, 100, 1000], 
              'gamma': ['scale','auto'],
              'kernel': ['rbf']} 
    grid = GridSearchCV(SVC(class_weight='balanced'), param_grid, refit = True, verbose = 3)
    grid.fit(X_train1A, y_train)
    print(grid.best_params_)
    print(grid.best_estimator_)
    grid_predictions = grid.predict(X_test1A)
    print(metrics.classification_report(y_test, grid_predictions))
