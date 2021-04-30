import logistic_model as lm
import neural_model as nm
import svm_model as svm
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
    #lm.LogisticModel(X_train, X_test, y_train, y_test, verbose=False)
    bag_models = [lm.LogisticModel(X_train, X_test, y_train, y_test, verbose=False), svm.svm(X_train, X_test, y_train, y_test), nm.NeuralModel(X_train, X_test, y_train, y_test)]
    y_list=[]
    for model in bag_models:
        model.build_model()
        y_list.append(model.predict(X_test))
    final_results = y_list[0]+y_list[1]+y_list[2]
    print(final_results)
    bag_y_predict = np.where(final_results<=1,0,1)
    print(bag_y_predict)
    #neural_ml = nm.NeuralModel(X_train, X_test, y_train, y_test)
    #neural_ml.build_model()
    #neural_y_predict = neural_ml.predict(X_test)

    # np.random.seed(167)
    # X_train, X_test_lm, y_train, y_test = train_test_split(stock_df.iloc[:,0:-1],stock_df.iloc[:,-1], test_size=0.2)
    # logistic_ml = lm.LogisticModel(X_train, X_test_lm, y_train, y_test)
    # logistic_ml.build_model()
    # logistic_y_predict = logistic_ml.predict(X_test)


    ### CALCULATE FINAL ACCURACY BASED ON PREDICTIONS FROM ALL MODELS
    #print(neural_y_predict)
    #print(logistic_y_predict)
    print("BAGGING MODEL ACCURACY", accuracy_score(y_test, bag_y_predict))



    ### Fucntion that compares models


if __name__ == "__main__":
    main()