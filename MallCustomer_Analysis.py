import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("Mall_Customers.csv")
print(data.head())

print("\n")

print(data.isnull().sum()) #Kayıp Veriler İncelendi.

del data["CustomerID"]
"""
data.hist()

plt.show()
"""
print("\n")

male = 0
female = 0

for i in data["Gender"]:
    if i == "Male":
        male += 1
    else:
        female += 1

print(f"Erkek Sayısı = {male}")
print(f"Kadın Sayısı = {female}")

"""
#data.groupby('Gender')['Spending Score (1-100)'].mean().plot(kind='bar')

plt.scatter(data["Annual Income (k$)"].values, data["Spending Score (1-100)"].values, color = 'red')
plt.title('AVM Müşteri İncelemesi')
plt.xlabel('Müşteri Maaşı (Yıllık $)')
plt.ylabel('Müşteri Skoru (1-100)')
"""

x = data[['Annual Income (k$)','Spending Score (1-100)']]

from sklearn.cluster import KMeans

wcss=[]

for i in range(1,11):
    kmeans = KMeans(i)
    kmeans.fit(x)
    wcss_iter = kmeans.inertia_
    wcss.append(wcss_iter)

number_clusters = range(1,11)

"""

plt.plot(number_clusters,wcss)
plt.title('The Elbow title')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')

"""


print("\n")

model = KMeans(n_clusters=5, random_state=42)

model.fit(x)

predicted = model.predict(x)

data["Tahminler"] = predicted

print(data)

"""

Visualize the Results

"""


plt.scatter(x.values[predicted==0,0],x.values[predicted==0,1],s=50, c='green',label='Cluster1')
plt.scatter(x.values[predicted==1,0],x.values[predicted==1,1],s=50, c='cyan',label='Cluster2')
plt.scatter(x.values[predicted==2,0],x.values[predicted==2,1],s=50, c='yellow',label='Cluster3')
plt.scatter(x.values[predicted==3,0],x.values[predicted==3,1],s=50, c='purple',label='Cluster4')
plt.scatter(x.values[predicted==4,0],x.values[predicted==4,1],s=50, c='blue',label='Cluster5')

plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1],s=200,marker='s', c='magenta', alpha=0.6, label='Centroids')
plt.title('Müşteri Kümeleri')
plt.xlabel('Müşterinin Yıllık Geliri')
plt.ylabel('Müşteri Puanı')
plt.legend()

plt.show()