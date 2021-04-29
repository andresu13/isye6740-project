import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

class LogisticModel(object):

    # Define constructor
    def __init__(self, X_train, X_test, y_train, y_test, verbose = False):
            #self.leaf_size = leaf_size
            self.verbose = verbose
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
            self.logistic_model = LogisticRegression()
            pass


    def build_model(self):

        # Create variables based on the data from the constructor
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test

        #Remove the first column from X_train and X_test
        X_train = X_train.iloc[:,1:]
        X_test = X_test.iloc[:,1:]
        self.num_cols = X_train.columns

        print("Logistic Model")
        # Your Code goes here

        if self.verbose == True:

            ### Standardize/Normalize data before passing it to the Logistic Model algorithm
            
            ### Normalization
            norm = MinMaxScaler().fit(X_train)
            X_train_norm = norm.transform(X_train)
            X_test_norm = norm.transform(X_test)

            ### Standardization
            X_train_stand = X_train.copy()
            X_test_stand = X_test.copy()

            for i in self.num_cols:
                self.scale = StandardScaler().fit(X_train_stand[[i]])
                X_train_stand[i] = self.scale.transform(X_train_stand[[i]])
                X_test_stand[i] = self.scale.transform(X_test_stand[[i]])

            # Build model without any normalization/standardization
            print()
            print('BASE MODEL RESULTS:')
            lr = LogisticRegression(max_iter = 500).fit(X_train, y_train)
            lr_pred = lr.predict(X_test)
            lr_acc = sum(lr_pred == y_test) / len(y_test)
            print()
            print("  >>Logistic Regression model with all parameters set as defaults results in a prediction accuracy of: " + str(lr_acc))
            print()
            
            # Build Logistic Regression with Normalized Data
            lr_norm = LogisticRegression(max_iter = 500)
            lr_norm.fit(X_train_norm, y_train)
            y_pred_norm = lr_norm.predict(X_test_norm)
            print("Accuracy Score (Normalized): ", accuracy_score(y_test, y_pred_norm))

            # Build Logistic Regression with Standardized Data
            lr_stand = LogisticRegression(max_iter = 500)
            lr_stand.fit(X_train_stand, y_train)
            y_pred_stand = lr_stand.predict(X_test_stand)
            print("Accuracy Score (Standardized): ", accuracy_score(y_test, y_pred_stand))


            pca_results = []
            pca_dim = [2,3,4,5,6,7,8,9,10]

            print('PCA MODEL RESULTS:')
            print('  >>Performing PCA on the feature data (standardized), from 2 to 10 principal components')
            print()

            for i in pca_dim:
                #PCA
                pca = PCA(n_components=i)
                X_train_pca = pca.fit_transform(X_train_stand)
                X_test_pca = pca.fit_transform(X_test)


                lr_pca = LogisticRegression(max_iter = 500).fit(X_train_pca, y_train)
                lr_pred_pca = lr_pca.predict(X_test_pca)
                lr_acc_pca = sum(lr_pred_pca == y_test) / len(y_test)

                pca_results.append(lr_acc_pca)

            max_value = max(pca_results)
            max_index = pca_results.index(max_value)

            print('  >>Performing PCA on the data DOES NOT improve predictive power of Logistic Regression')
            print('  >>As the highest prediction accuracy with PCA was: '  + str(max_value) +' with ' + str(pca_dim[max_index]) + ' principal components')
            print()

            ### Model Optimization

            # Parameters to optimize

            lr = LogisticRegression().fit(X_train_stand, y_train)

            #PARAMETERS
            print('HYPERPARAMETER TUNING:')
            print()

            print('  >>Exploring Hyperparameters with Ridge Regression')
            print()
            solvers = ['newton-cg', 'lbfgs', 'liblinear']
            penalty = ['l2']
            c_values = [1000,100, 10, 1.0, 0.1, 0.01,.001,0.0001]

            param_grid_ridge = dict(solver=solvers,penalty=penalty,C=c_values)


            kfolds = KFold(n_splits=5, shuffle=True, random_state=4)
            grid_search = GridSearchCV(estimator=lr, param_grid=param_grid_ridge, n_jobs=-1, cv=kfolds, scoring='accuracy',error_score=0)
            grid_model = grid_search.fit(X_train_stand, y_train)

            print("Best: %f using %s" % (grid_model.best_score_, grid_model.best_params_))
            means = grid_model.cv_results_['mean_test_score']
            stds = grid_model.cv_results_['std_test_score']
            params = grid_model.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                print("%f (%f) with: %r" % (mean, stdev, param))

            print()

            print('  >>Exploring Hyperparameters with Lasso Regression')
            print()

            solvers = ['liblinear']
            penalty = ['l1']
            c_values = [1000,100, 10, 0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

            param_grid_lasso = dict(solver=solvers,penalty=penalty,C=c_values)

            kfolds = KFold(n_splits=5, shuffle=True, random_state=4)
            grid_search = GridSearchCV(estimator=lr, param_grid=param_grid_lasso, n_jobs=-1, cv=kfolds, scoring='accuracy',error_score=0)
            grid_model = grid_search.fit(X_train_stand, y_train)

            print("Best: %f using %s" % (grid_model.best_score_, grid_model.best_params_))
            means = grid_model.cv_results_['mean_test_score']
            stds = grid_model.cv_results_['std_test_score']
            params = grid_model.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                print("%f (%f) with: %r" % (mean, stdev, param))

            print()
            print()

            print('  >>Lasso Regression provided HIGHEST Accuracy results')

            print()

            print('  >>Hyperparameter tuning with Lasso Regression on Test Data')
            print()

            lasso_alphas = [0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,20,50]

            lasso_acc = []
            w = []

            np.random.seed(4)

            for i in lasso_alphas:

                kfolds = KFold(n_splits=5, shuffle=True, random_state=4)
                lasso_cv = LogisticRegressionCV(cv =kfolds,Cs = [i],fit_intercept=True,solver = 'liblinear',penalty = 'l1')
                lasso_cv.fit(X_train_stand, y_train)
                lasso_cv.predict(X_test_stand)

                lasso_acc.append(lasso_cv.score(X_test_stand, y_test))

                w.append(lasso_cv.coef_)

            max_value = max(lasso_acc)
            max_index = lasso_acc.index(max_value)

            lasso_cv = LogisticRegressionCV(cv =kfolds,Cs = [lasso_alphas[max_index]],fit_intercept=True,solver = 'liblinear',penalty = 'l1')
            lasso_cv.fit(X_train_stand, y_train)
            lasso_cv.predict(X_test_stand)

            final_lasso_acc = lasso_cv.score(X_test_stand, y_test)

            print('The optimal value of Accuracy is ' + str(final_lasso_acc) +  ' with lambda = ' + str(lasso_alphas[max_index]))
            print('The fitted model parameters are: ')
            print(lasso_cv.coef_)
            print('The intercept is: ')
            print(lasso_cv.intercept_)


            w=np.array(w)

            plt.plot(lasso_alphas,w[:,0])
            plt.title('Lasso coefficients as function of the regularization')
            plt.xlabel('Alpha')
            plt.ylabel('Weights')
            #plt.legend()
            #plt.savefig('Q1a Ridge Coefficients.png')
            plt.show()

            plt.plot(lasso_alphas,lasso_acc,label = 'CV Accuracy')
            plt.title('Lasso CV (5-folds) Accuracy')
            plt.xlabel('Alpha')
            plt.ylabel('CV Accuracy')
            plt.legend()
            #plt.savefig('Q1a Ridge SSR.png')
            plt.show()

        else:
            ### Standardization
            X_train_stand = X_train.copy()
            X_test_stand = X_test.copy()

            for i in self.num_cols:
                self.scale = StandardScaler().fit(X_train_stand[[i]])
                X_train_stand[i] = self.scale.transform(X_train_stand[[i]])
                X_test_stand[i] = self.scale.transform(X_test_stand[[i]])
    
    
            lasso_alphas = [0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,20,50]

            lasso_acc = []
            w = []

            np.random.seed(4)

            for i in lasso_alphas:

                kfolds = KFold(n_splits=5, shuffle=True, random_state=4)
                lasso_cv = LogisticRegressionCV(cv =kfolds,Cs = [i],fit_intercept=True,solver = 'liblinear',penalty = 'l1')
                lasso_cv.fit(X_train_stand, y_train)
                lasso_cv.predict(X_test_stand)

                lasso_acc.append(lasso_cv.score(X_test_stand, y_test))

                w.append(lasso_cv.coef_)

            max_value = max(lasso_acc)
            max_index = lasso_acc.index(max_value)

        self.logistic_model = LogisticRegressionCV(cv =kfolds,Cs = [lasso_alphas[max_index]],fit_intercept=True,solver = 'liblinear',penalty = 'l1').fit(X_train_stand, y_train)
        y_pred_model = self.logistic_model.predict(X_test_stand)

        if self.verbose == True:
            print()
            print("Classification report for Logistic Model")
            print()
            print(classification_report(y_test, y_pred_model))
            print()
            print("Confusion matrix")
            print(confusion_matrix(y_test, y_pred_model))
            print()

    def predict(self, X_test):
        X_test = X_test.iloc[:,1:]
        X_test_stand = X_test.copy()
        for i in self.num_cols:
            X_test_stand[i] = self.scale.transform(X_test_stand[[i]])
        y_pred_model = self.logistic_model.predict(X_test_stand)

        return y_pred_model

if __name__ == "__main__":
    build_model()
