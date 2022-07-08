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

stocks = int(input("Number owned: "))

spent = float(input("Bought for: ")) * stocks

data['profit'] = (data['close'] * stocks) - spent

print(spent)

print(data.head())



def cleanData(dataSet):
    dataSet['SMA5'].fillna(dataSet['close'], inplace=True)
    return dataSet

def predictClose(dataSet):
    dataNew = cleanData(dataSet[['SMA5', 'close', 'volume']])
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
    

#plt.scatter(x=data['volume'], y=data['close'])

#plt.show()

predictClose(data)
plt.plot(data['timestamp'], data['close'])
plt.xticks(rotation=45, fontsize=6)
ax = plt.gca()
#ax.axes.get_xaxis().set_ticks()
xticks = ax.get_xticklabels()
labels = []
for x in range(0, len(data['timestamp'])):
    if(x == 0 or x == len(data['timestamp']) - 1):
        labels.append(data['timestamp'][x])
    else:
        labels.append("")
ax.axes.get_xaxis().set_ticks(labels)
ax.invert_xaxis()
#ax.axes.get_xaxis().set_ticks(data['timestamp'][len(data['timestamp']) - 1])
data["SMA5"].fillna(data['close'], inplace=True)
plt.plot(data['timestamp'], data['SMA5'])
plt.axhline(y=spent/stocks, color='r')
plt.show()