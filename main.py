import matplotlib.pyplot as plt
import requests
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle


# @hidden_cell
key = '9HIHFNRD0J1APKU9&datatype=csv'

equity = input('Equity symbol: ')

url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=' + equity + '&apikey=' + key
r = requests.get(url)
data = r.content
csv_file = open('data.csv', 'wb')

csv_file.write(data)
csv_file.close()

data = pd.read_csv('data.csv')

data['change'] = data['close'] - data['open']

data['SMA5'] = data['close'].rolling(5).mean()

print(data.head())


def cleanData(dataSet):
    dataNew = dataSet.dropna()
    return dataNew

def predictClose(dataSet):
    dataNew = cleanData(dataSet[['SMA5', 'close']])
    print(dataNew.head())
    predict = 'close'
    X = np.array(dataNew.drop([predict], axis=1))
    y = np.array(dataNew[predict])
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)
    
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)

    print("Coefficient: \n", linear.coef_)
    print("Intercept: \n", linear.intercept_)

    predictions = linear.predict(x_test)

    for x in range(len(predictions)):
        print(predictions[x], x_test[x], y_test[x], y_test[x] - predictions[x])
    

"""data.plot(x ="timestamp", y = "close")
plt.scatter(x=data["open"], y = data["close"])
plt.show()"""

predictClose(data)
