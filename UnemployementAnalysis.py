import pandas as pd
import numpy as np

data = pd.read_csv("unemployment_analysis.csv")

del data["Country Name"]

print(data)

print(data.iloc[216:217])

new_data = data.iloc[216:217]

print(new_data)

del new_data["Country Code"]

x = new_data.iloc[0].values.reshape(-1,1)

print(x)

y = np.array([1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021]).reshape(-1,1)

print(y)

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
prediction = regressor.predict(X_train)
print(prediction)


import matplotlib.pyplot as plt

plt.scatter(x, y, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('İşsizlik Oranı Türkiye')
plt.xlabel('İşsizlik Oranı')
plt.ylabel('Yıl')
plt.show()


