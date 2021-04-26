import urllib.request
import json
import os
import ssl
import pandas as pd

def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

data = {
    "data":
    [
        {
            'Column2': "example_value",
            'bookValuePerShare': 0,
            'cashAndEquivalents': "0",
            'debtToEquityRatio': "0",
            'earningsPerBasicShare': "0",
            'freeCashFlowPerShare': "0",
            'grossProfit': "0",
            'netCashFlow': "0",
            'netIncome': "0",
            'operatingExpenses': "0",
            'priceSales': "0",
            'researchAndDevelopmentExpense': "0",
            'RD_missing': "0",
            'revenues': "0",
            'salesPerShare': "0",
            'totalLiabilities': "0",
            'profitMargin': "0",
        },
    ],
}

print(data)
test_df =  pd.read_csv("sample_stock_data_test.csv")
test_df = test_df.astype('object')
print(test_df.dtypes)

# Code below for an individual row
#data = {"data":[test_df.iloc[21,1:-1].to_dict()]}

# Code below if you want to send the entire test dataset
data = {"data":test_df.iloc[:,1:-1].to_dict('records')}
print(data)
body = str.encode(json.dumps(data))

url = 'http://ee3f9805-2ff0-4386-b31b-75d36cd4b9fd.eastus.azurecontainer.io/score'
api_key = '' # Replace this with the API key for the web service
headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

req = urllib.request.Request(url, body, headers)

try:
    response = urllib.request.urlopen(req)

    result = response.read()
    print(result)
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())
    print(json.loads(error.read().decode("utf8", 'ignore')))