import logistic_model
import neural_model as nm
#import svm_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    #np.random.seed(30)
    stock_df = pd.read_csv("sample_stock_data.csv")
    stock_df = stock_df.drop("priceSales", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(stock_df.iloc[:,0:-1],stock_df.iloc[:,-1], test_size=0.2)
    #print(X_train)

    
    #X_train.to_csv("sample_stock_data_train.csv")
    #y_train.to_csv("sample_stock_label_train.csv")
    #X_test.to_csv("sample_stock_data_test.csv")
    #y_test.to_csv("sample_stock_label_test.csv")

    ### These 3 lines will be called once the models are built
    ### Expected API Return: Accuracy, Recall, F1 Score

    ### Build models
    #log_ml = logistic_model.build_model(X_train, X_test, y_train, y_test )
    neural_ml = nm.NeuralModel(X_train, X_test, y_train, y_test)
    neural_ml.build_model()
    print(neural_ml.predict(X_test))
    #neural_ml = neural_model.build_model(X_train, X_test, y_train, y_test)
    #svm_ml = svm_model.build_model(X_train, X_test, y_train, y_test )


    ### Fucntion that compares models


if __name__ == "__main__":
    main()