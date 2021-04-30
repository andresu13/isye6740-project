import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.metrics import accuracy_score

class svm(object):  	

    # Define constructor
    def __init__(self, X_train, X_test, y_train, y_test, verbose = False):
            #self.leaf_size = leaf_size
            self.verbose = verbose
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
            self.svm = SVC()
            pass

    def build_model_local(self, X_train, X_test, y_train, y_test,safe=False):
        print("Grid Search SVM Model")
        #print(X_train.head())
        #print('train shape: ',X_train.shape)
        #print('test shape: ',X_test.shape)

        # Hyperparameter tuning 
        param_grid = {'C': [0.01, 0.1, 1, 10, 100, 1000], 
                'gamma': ['scale','auto'],
                'kernel': ['rbf','poly','sigmoid']}
        if safe == True:
            self.svm = GridSearchCV(SVC(), param_grid, refit = True, verbose = 0,scoring='accuracy')
            self.svm.fit(X_train, y_train)
            print(self.svm.best_params_)
            print(self.svm.best_estimator_)
            grid_predictions = self.svm.predict(X_test)
            print('accuracy: ', metrics.classification_report(y_test, grid_predictions,output_dict=True)['accuracy'])
            print(metrics.classification_report(y_test, grid_predictions))
            print(metrics.confusion_matrix(y_test, grid_predictions))
            best_params = self.svm.best_params_
        if safe == False:
            grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 0,scoring='accuracy')
            grid.fit(X_train, y_train)
            print(grid.best_params_)
            print(grid.best_estimator_)
            grid_predictions = grid.predict(X_test)
            print('accuracy: ', metrics.classification_report(y_test, grid_predictions,output_dict=True)['accuracy'])
            print(metrics.classification_report(y_test, grid_predictions))
            print(metrics.confusion_matrix(y_test, grid_predictions))
            best_params = grid.best_params_
        #metrics.plot_confusion_matrix(grid, X_test, y_test)
        #plt.savefig("SVM_Confusion_Matrix")
        return best_params
    
    def build_svm(self, X_train, X_test, y_train, y_test, gamma, kernel):
        print("SVM Model")
        # Your Code goes here
        # print(X_train.head())
        #print('train shape: ',X_train.shape)
        #print('test shape: ',X_test.shape)
        #Cs = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10 ,100, 1000, 10000, 100000]
        Cs = [1e-8,1e-7,1e-6,1e-5,1e-5,1e-4,1e-3,1e-2,1e-1,1,1e1,1e2,1e3,1e4,1e5,]
        acc = []
        for C in Cs: 
            svm = SVC(C= C, gamma='scale', kernel = 'rbf',class_weight='balanced',random_state=42).fit(X_train,y_train)
            y_pred = svm.predict(X_test)
            acc.append(accuracy_score(y_test, y_pred))
            print("Accuracy Score (Standardized): ", accuracy_score(y_test, y_pred))
        plt.figure()
        plt.plot(Cs,acc)
        plt.xscale('log')
        plt.xlabel('C')
        plt.ylabel('accuracy')
        plt.savefig("SVM_C_acc")
    
    def build_model(self):

        # Create variables based on the data from the constructor
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test

        #Remove the first column from X_train and X_test
        self.X_train1 = X_train.iloc[:,1:]
        X_train1 = self.X_train1
        X_test1 = X_test.iloc[:,1:]
        
        print("SVM MODEL")

        # Original Data
        print('### Original Data ###')
        best_params = self.build_model_local(X_train1, X_test1, y_train, y_test)

        # Normalize data
        print('### Normalize Data ###')
        X_train1A = (X_train1-X_train1.min())/(X_train1.max()-X_train1.min())
        X_test1A = (X_test1-X_train1.min())/(X_train1.max()-X_train1.min())
        best_params_norm = self.build_model_local(X_train1A, X_test1A, y_train, y_test,safe=True)

        # Standararize data
        print('### Standarize Data ###')
        X_train1B = (X_train1-X_train1.mean())/X_train1.std()
        X_test1B = (X_test1-X_train1.mean())/X_train1.std()
        best_params_norm = self.build_model_local(X_train1B, X_test1B, y_train, y_test)

        # Visualizing variance explained in PCA to select PCs
        pca = PCA(n_components=len(X_train1.columns))
        pc = pca.fit_transform(X_train1B)
        print(pca.explained_variance_ratio_)
        df = pd.DataFrame({'var':pca.explained_variance_ratio_,
                'PC':list(range(1,16))})
        ax = sns.barplot(x='PC',y="var", data=df, color="c")
        ax.set(xlabel='Principal Components', ylabel='Explained Variance')
        plt.savefig("PCA_scree_plot")

        # Selecting 2 Principal Components and then visualizing clusters
        plt.figure()
        pca = PCA(n_components=2)
        pc = pca.fit_transform(X_train1B)
        pc_df = pd.DataFrame(data = pc , 
            columns = ['PC1', 'PC2'])
        pc_df['Cluster'] = y_train
        sns.lmplot( x="PC1", y="PC2",
            data=pc_df, 
            fit_reg=False, 
            hue='Cluster', # color by cluster
            legend=True,
            scatter_kws={"s": 20}) # specify the point size
        plt.savefig("PCs_clusters")

        if self.verbose == True:
            # PCA with 3 components 
            print('### PCA with 3 components ###')
            pca = PCA(n_components=4)
            X_pca_train = pca.fit_transform(X_train1B)
            X_pca_test = pca.transform(X_test1B)
            self.build_model_local(X_pca_train, X_pca_test, y_train, y_test)

        # Plot vs accuracy 
        print(best_params_norm)
        gamma = best_params_norm['gamma']
        kernel = best_params_norm['kernel']
        print(gamma)
        print(kernel)
        self.build_svm(X_train1A, X_test1A, y_train, y_test,gamma,kernel)
    
    def predict(self,X_test):
        X_test1 = X_test.iloc[:,1:]
        X_test1A = (X_test1-self.X_train1.min())/(self.X_train1.max()-self.X_train1.min())
        return self.svm.predict(X_test1A)



def build_model(X_train, X_test, y_train, y_test):
    print("Grid Search SVM Model")
    #print(X_train.head())
    #print('train shape: ',X_train.shape)
    #print('test shape: ',X_test.shape)

    # Hyperparameter tuning 
    param_grid = {'C': [0.01, 0.1, 1, 10, 100, 1000], 
              'gamma': ['scale','auto'],
              'kernel': ['rbf','poly','sigmoid']} 
    grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 0,scoring='accuracy')
    grid.fit(X_train, y_train)
    print(grid.best_params_)
    print(grid.best_estimator_)
    grid_predictions = grid.predict(X_test)
    print('accuracy: ', metrics.classification_report(y_test, grid_predictions,output_dict=True)['accuracy'])
    print(metrics.classification_report(y_test, grid_predictions))
    print(metrics.confusion_matrix(y_test, grid_predictions))
    #metrics.plot_confusion_matrix(grid, X_test, y_test)
    #plt.savefig("SVM_Confusion_Matrix")
    return grid.best_params_

def build_svm(X_train, X_test, y_train, y_test, gamma, kernel):
    print("SVM Model")
    # Your Code goes here
    # print(X_train.head())
    #print('train shape: ',X_train.shape)
    #print('test shape: ',X_test.shape)
    #Cs = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10 ,100, 1000, 10000, 100000]
    Cs = [1e-8,1e-7,1e-6,1e-5,1e-5,1e-4,1e-3,1e-2,1e-1,1,1e1,1e2,1e3,1e4,1e5,1e6,1e7]
    acc = []
    for C in Cs: 
        svm = SVC(C= C, gamma='scale', kernel = 'rbf',class_weight='balanced',random_state=42).fit(X_train,y_train)
        y_pred = svm.predict(X_test)
        acc.append(accuracy_score(y_test, y_pred))
        print("Accuracy Score (Standardized): ", accuracy_score(y_test, y_pred))
    plt.figure()
    plt.plot(Cs,acc)
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('accuracy')
    plt.savefig("SVM_C_acc")


if __name__ == "__main__":
    np.random.seed()
    stock_df = pd.read_csv("sample_stock_data.csv")
    stock_df = stock_df.drop("priceSales", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(stock_df.iloc[:,0:-1],stock_df.iloc[:,-1], test_size=0.2)

    svm_ml = svm(X_train, X_test, y_train, y_test)
    svm_ml.build_model()
    svm_predict = svm_ml.predict(X_test)
