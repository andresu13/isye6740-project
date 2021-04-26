import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

def build_model(X_train, X_test, y_train, y_test ):
    #Remove the first column from X_train and X_test
    X_train = X_train.iloc[:,1:]
    X_test = X_test.iloc[:,1:]
    num_cols = X_train.drop("RD_missing", axis=1).columns
    #num_cols = ['Item_Weight','Item_Visibility','Item_MRP','Outlet_Establishment_Year']
    #RD_missing
    print("Neural Model")
    #print(X_train.head())
    #print(X_train.describe())

    ### Standardize/Normalize data before passing it to the Neural Network algorithm

    ### Normalization
    norm = MinMaxScaler().fit(X_train)
    X_train_norm = norm.transform(X_train)
    X_test_norm = norm.transform(X_test)

    pca_norm = PCA(n_components=10)
    X_pca_train_norm = pca_norm.fit_transform(X_train_norm)
    X_pca_test_norm = pca_norm.transform(X_test_norm)
    #print(X_train_norm[0:10,])

    ### Standardization

    X_train_stand = X_train.copy()
    X_test_stand = X_test.copy()

    for i in num_cols:
        scale = StandardScaler().fit(X_train_stand[[i]])
        X_train_stand[i] = scale.transform(X_train_stand[[i]])
        X_test_stand[i] = scale.transform(X_test_stand[[i]])

    pca_stand = PCA(n_components=10)
    X_pca_train_stand = pca_stand.fit_transform(X_train_stand)
    X_pca_test_stand = pca_stand.transform(X_test_stand)
    

    # RD_missing

    #### Run Neural Networks Classifier

    #clf = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500, alpha=0.0001,solver='sgd', verbose=10,  random_state=21,tol=0.000000001)
    
    # Build Neural Network with Normalized Data
    clf_norm = MLPClassifier()
    clf_norm.fit(X_train_norm, y_train)
    y_pred_norm = clf_norm.predict(X_test_norm)
    print("Accuracy Score (Normalized): ", accuracy_score(y_test, y_pred_norm))

    # Build Neural Network with Normalized Data (after PCA)
    clf_pca_norm = MLPClassifier()
    clf_pca_norm.fit(X_pca_train_norm, y_train)
    y_pred_pca_norm = clf_pca_norm.predict(X_pca_test_norm)
    print("Accuracy Score (Normalized and PCA): ", accuracy_score(y_test, y_pred_pca_norm))

    # Build Neural Network with Standardized Data
    clf_stand = MLPClassifier()
    clf_stand.fit(X_train_stand, y_train)
    y_pred_stand = clf_stand.predict(X_test_stand)
    print("Accuracy Score (Standardized): ", accuracy_score(y_test, y_pred_stand))

    # Build Neural Network with Standardized Data (after PCA)
    clf_pca_stand = MLPClassifier()
    clf_pca_stand.fit(X_pca_train_stand, y_train)
    y_pred_pca_stand = clf_pca_stand.predict(X_pca_test_stand)
    print("Accuracy Score (Standardized and PCA): ", accuracy_score(y_test, y_pred_pca_stand))



    ### Model Optimization

    # Parameters to optimize

    clf_gs = MLPClassifier()

    # To choose the number of neurons in the hidden layer, I used the following rule of thumb and tried all of them using GridSearchCV:
    # - Mean between number of neurons in input layer and output layer
    # - 2/3 the size of the input layer plut the size of the output layer
    # - Less than twice the size of the input layer

    parameters = {
        #'hidden_layer_sizes': [(6,),(9,),(15,)],
        'activation': ['tanh', 'relu', 'logistic'],
        'solver': ['lbfgs','sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive', 'invscaling']
    }
    kfolds =KFold(n_splits=5, shuffle=True, random_state=4)
    grid_search = GridSearchCV(clf_gs, parameters, n_jobs=-1, cv=kfolds, scoring='accuracy', error_score=0)
    grid_model = grid_search.fit(X_pca_train_stand,y_train)

    print("Best: %f using %s" % (grid_model.best_score_, grid_model.best_params_))

    y_pred_grid_model = grid_model.predict(X_pca_test_stand)
    print("Accuracy Score (after Grid Search): ", accuracy_score(y_test, y_pred_grid_model))

if __name__ == "__main__":
    build_model()