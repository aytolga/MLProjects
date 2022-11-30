#!/usr/bin/env python
# coding: utf-8

# In[71]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[72]:


data = pd.read_csv("global life expectancy dataset.csv")
data.head()


# In[73]:


#We are going to find USA and Turkey Datas.

tr_data = 0
usa_data = 0

for i in data["Country Code"]:
    tr_data += 1
    if i == "TUR":
        break
for j in data["Country Code"]:
    usa_data += 1
    if j == "USA":
        break
print(f"Turkey in {tr_data-1}")
print(f"USA in {usa_data-1}")


# In[74]:


del data["Country Name"]
del data["Country Code"]


# In[75]:


tr = data.iloc[187]
usa = data.iloc[192]


# In[76]:


tr


# In[77]:


usa


# In[78]:


years = np.arange(1960,2021).reshape(-1,1)


# In[79]:


tr_array = np.array(tr).reshape(-1,1)


# In[80]:


usa_array = np.array(usa).reshape(-1,1)


# In[81]:


plt.plot(tr,years,color="r")
plt.plot(usa,years,color="b")

plt.xlabel("The Life Expectancy")
plt.ylabel("Years")

plt.legend(["Turkey","USA"])


# In[82]:


figure1 = plt.bar(60, tr[0], align='edge', width=0.4, label='Turkey')
figure2 = plt.bar(60, usa[0], align='edge', width=-0.4, label='USA')
plt.title("Life Expectancy in 1960")
plt.legend(["Turkey","USA"])


# In[83]:


figure1 = plt.bar(60, tr[60], align='edge', width=0.4, label='Turkey')
figure2 = plt.bar(60, usa[60], align='edge', width=-0.4, label='USA')
plt.title("Life Expectancy in 2020")
plt.legend(["Turkey","USA"])


# In[84]:



from sklearn.model_selection import train_test_split
x_tr_train, x_tr_test, y_tr_train, y_tr_test = train_test_split(tr_array, years, test_size=0.3, random_state=0)
x_usa_train, x_usa_test, y_usa_train, y_usa_test = train_test_split(usa_array, years, test_size=0.3, random_state=0)


# In[85]:


#Regression Analysis

from sklearn.linear_model import LinearRegression
lr_tr = LinearRegression()
lr_tr.fit(x_tr_train,y_tr_train)
tr_sq = lr_tr.score(x_tr_test, y_tr_test)
print(f"coefficient of determination: {tr_sq}")


# In[86]:


#Regression Analysis

from sklearn.linear_model import LinearRegression
lr_usa = LinearRegression()
lr_usa.fit(x_usa_train,y_usa_train)
usa_sq = lr_usa.score(x_usa_test, y_usa_test)
print(f"coefficient of determination: {usa_sq}")


# In[87]:


#Graphs For Regression

tr_pred = lr_tr.predict(x_tr_test)
usa_pred = lr_usa.predict(x_usa_test)


# In[88]:


plt.scatter(x_tr_train,y_tr_train,color="r")
plt.plot(x_tr_test,tr_pred,color="b")

plt.title("Turkey's datas with Linear Regression")
plt.xlabel("Life Expectancy Rate")
plt.ylabel("Years")


# In[89]:


plt.scatter(x_usa_train,y_usa_train,color="r")
plt.plot(x_usa_test,usa_pred,color="b")

plt.title("USA's datas with Linear Regression")
plt.xlabel("Life Expectancy Rate")
plt.ylabel("Years")


# In[148]:


#Polynomial Regression

from sklearn.preprocessing import PolynomialFeatures

poly1 = PolynomialFeatures(2)

poly_fit_tr = poly1.fit_transform(tr_array)

tr_poly = LinearRegression()

tr_poly.fit(poly_fit_tr,years)

tr_pred = tr_poly.predict(poly_fit_tr)


# In[153]:


plt.scatter(tr,years,color="r",alpha=0.2)
plt.plot(tr,tr_pred)
plt.xlabel("Expectancy Rate")
plt.ylabel("Years")
plt.title("Turkey")


# In[160]:


poly2 = PolynomialFeatures(3)

poly_fit_usa = poly1.fit_transform(usa_array)

usa_poly = LinearRegression()

usa_poly.fit(poly_fit_usa,years)

usa_pred = usa_poly.predict(poly_fit_usa)


# In[161]:


plt.scatter(usa,years,color="r",alpha=0.2)
plt.plot(usa,usa_pred)
plt.xlabel("Expectancy Rate")
plt.ylabel("Years")
plt.title("USA")


# In[ ]:


#End Of the Analysis and Comperasion!

