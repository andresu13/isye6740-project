import logistic_model
import neural_model as nm
#import svm_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    #np.random.seed(30)
    stock_df = pd.read_csv("sample_stock_data.csv")
    stock_df = stock_df.drop("priceSales", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(stock_df.iloc[:,0:-1],stock_df.iloc[:,-1], test_size=0.2)
    print(X_train)
    print(y_train)

    ### These 3 lines will be called once the models are built
    ### Expected API Return: Accuracy, Recall, F1 Score

    ### Build models
    #log_ml = logistic_model.build_model(X_train, X_test, y_train, y_test )

    ### BUILD EACH MODEL AND GET PREDICTIONS FOR EACH MODEL
    neural_ml = nm.NeuralModel(X_train, X_test, y_train, y_test)
    neural_ml.build_model()
    neural_y_predict = neural_ml.predict(X_test)


    ### CALCULATE FINAL ACCURACY BASED ON PREDICTIONS FROM ALL MODELS
    print(neural_y_predict)
    print("BAGGING MODEL ACCURACY", accuracy_score(y_test, neural_y_predict))



    ### Fucntion that compares models


if __name__ == "__main__":
    main()