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
    Cs = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10 ,100, 1000, 10000, 100000]
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
    X_train1 = X_train.drop(['Unnamed: 0'],axis=1)
    X_test1 = X_test.drop(['Unnamed: 0'],axis=1)

    # Original Data
    best_params = build_model(X_train1, X_test1, y_train, y_test)

    # Normalize data
    X_train1A = (X_train1-X_train1.min())/(X_train1.max()-X_train1.min())
    X_test1A = (X_test1-X_train1.min())/(X_train1.max()-X_train1.min())
    best_params_norm = build_model(X_train1A, X_test1A, y_train, y_test)

    # Standararize data
    X_train1B = (X_train1-X_train1.mean())/X_train1.std()
    X_test1B = (X_test1-X_train1.mean())/X_train1.std()
    best_params_norm = build_model(X_train1B, X_test1B, y_train, y_test)

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

    # # PCA with 3 components 
    # print('PCA with 3 components')
    # pca = PCA(n_components=4)
    # X_pca_train = pca.fit_transform(X_train1B)
    # X_pca_test = pca.transform(X_test1B)
    # build_model(X_pca_train, X_pca_test, y_train, y_test)

    # Plot vs accuracy 
    print(best_params_norm)
    gamma = best_params_norm['gamma']
    kernel = best_params_norm['kernel']
    print(gamma)
    print(kernel)
    build_svm(X_train1A, X_test1A, y_train, y_test,gamma,kernel)

