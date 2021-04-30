import logistic_model as lm
import neural_model as nm
import svm_model as svm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    np.random.seed(30)
    stock_df = pd.read_csv("sample_stock_data.csv")
    stock_df = stock_df.drop("priceSales", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(stock_df.iloc[:,0:-1],stock_df.iloc[:,-1], test_size=0.2)
    print(X_train)
    print(y_train)


    ### BUILD EACH MODEL AND GET PREDICTIONS FROM EACH MODEL TO BE USED BY BAGGING ALGORITHM

    bag_models = [lm.LogisticModel(X_train, X_test, y_train, y_test, verbose=False), svm.svm(X_train, X_test, y_train, y_test), nm.NeuralModel(X_train, X_test, y_train, y_test, verbose=False)]
    y_list=[]
    for model in bag_models:
        model.build_model()
        y_list.append(model.predict(X_test))
    final_results = y_list[0]+y_list[1]+y_list[2]
    print(final_results)
    bag_y_predict = np.where(final_results<=1,0,1)
    print(bag_y_predict)

    print("BAGGING MODEL ACCURACY", accuracy_score(y_test, bag_y_predict))

    #test_lm = lm.LogisticModel(X_train, X_test, y_train, y_test, verbose=True)
    #test_lm.build_model()

    #test_nm = nm.NeuralModel(X_train, X_test, y_train, y_test, verbose=True)
    #test_nm.build_model()

if __name__ == "__main__":
    main()