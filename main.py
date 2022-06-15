from parso import split_lines
import requests
import pandas as pd

# @hidden_cell



url = 'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol=IBM&apikey=' + key
r = requests.get(url)
data = r.content
csv_file = open('data.csv', 'wb')

csv_file.write(data)
csv_file.close()

data = pd.read_csv('data.csv')
print(data.head())

