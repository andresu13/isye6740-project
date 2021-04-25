import logistic_model
#import neural_model
#import svm_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def main():
    np.random.seed(30)
    stock_df = pd.read_csv("sample_stock_data.csv")

    #Standardizing Data
    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(stock_df.iloc[:,1:-1])

    X_train, X_test, y_train, y_test = train_test_split(X,stock_df.iloc[:,-1], test_size=0.8)
    #print(X_train)
    print(X_train.head())

    ### These 3 lines will be called once the models are built
    ### Expected API Return: Accuracy, Recall, F1 Score

    #log_ml = logistic_model.build_model(X_train, X_test, y_train, y_test )
    #neural_ml = neural_model.build_model(X_train, X_test, y_train, y_test )
    #svm_ml = svm_model.build_model(X_train, X_test, y_train, y_test )


    ### Fucntion that compares models


if __name__ == "__main__":
    main()