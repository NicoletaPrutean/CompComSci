import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler    

df = pd.read_csv("https://cssbook.net/d/eurobarom_nov_2017.csv")\
                 .dropna()\
                 .groupby(["country"])[[
                 "support_refugees_n",
                 "support_migrants_n",
                 "age",
                 "educational_n"]].mean()


#standardize data to z-scores so varibales with larger ranges do not dominate
scaler = StandardScaler()
df_s = scaler.fit_transform(df)

mypca = PCA()
componentscore = mypca.fit_transform(df_s)
scores_df = pd.DataFrame(componentscore, index=df.index)

#PCA loadings tell us how much each variable contributes to each component
#the loadings are the correlation between the original variables and the components
components_df = pd.DataFrame(data=mypca.components_.T) #transpose to have variables as rows and components as columns
components_df.index = df.columns
print("Variable loadings on the 4 components:")
print(components_df)

print("The component scores for each case=country:")
print(scores_df)


#plot countires in terms of 2 components
#arrows show the contribution of each variable to the components i.e., age, education to 
import bioinfokit.visuz
bioinfokit.visuz.cluster.biplot(cscore=componentscore,
                                loadings=mypca.components_,
                                labels=df.columns,
                                var1=round(mypca.explained_variance_ratio_[0], 2),
                                var2=round(mypca.explained_variance_ratio_[1], 2),
                                show=True
                                )

#PC reveals that in some countries older and more educated people are more likely to support refugees and migrants, while in other countries the opposite is true.
#This is a good example of how PCA can reveal complex relationships in data that are not immediately obvious from the raw data.

#let's visualize the other components as well, but it explains very littles variance
bioinfokit.visuz.cluster.biplot(cscore=componentscore,
                                loadings=mypca.components_,
                                labels=df.columns,
                                var1=round(mypca.explained_variance_ratio_[2], 2),
                                var2=round(mypca.explained_variance_ratio_[3], 2),
                                show=True
                                )   


#home exercise
#open a csv file from github with the following url: https://github.com/uvacw/teaching-bdaca/blob/main/12ec-course/week04/exercises/airbnb-datafile.csv, under name abnb

abnb = pd.read_csv("https://raw.githubusercontent.com/uvacw/teaching-bdaca/refs/heads/main/12ec-course/week04/exercises/airbnb-datafile.csv")

#explore data and clean up a bit
abnb.drop(columns=[ "Unnamed: 0", "survey_id", "host_id", "room_id", "room_type", "city", "country", "borough", "neighborhood", "minstay", "bathrooms", "name", "last_modified", "location"], inplace=True) #drop index column and id column

#transform some columns to numeric
#abnb["room_type"] = (abnb["room_type"] != "Private room").astype(int) #assigns 1 to entire apartment and 0 to private room

#unsupervised ML: group apartments based on some features I select
#run PCA to reduce number of features

abnb.dropna(inplace=True)  # drop rows with missing values
scaler = StandardScaler()

#standardize some features  
#abnb_s = scaler.fit_transform(abnb[["reviews", "overall_satisfaction", "accommodates", "bedrooms", "price", "latitude", "longitude"]])
abnb_s = scaler.fit_transform(abnb)

#assign the standardized features back to the dataframe
abnb[["reviews", "overall_satisfaction", "accommodates", "bedrooms", "price", "latitude", "longitude"]] = abnb_s

mypca_abnb = PCA()
componentscore_abnb = mypca_abnb.fit_transform(abnb_s)

scores_df_abnb = pd.DataFrame(componentscore_abnb, index=abnb.index)

#PCA loadings tell us how much each variable contributes to each component
#the loadings are the correlation between the original variables and the components

components_df_abnb = pd.DataFrame(data=mypca_abnb.components_.T)  # transpose to have variables as rows and components as columns

components_df_abnb.index = abnb.columns
print("Variable loadings on the components for Airbnb data:")
print(components_df_abnb)
print("The component scores for each case=apartment:")
print(scores_df_abnb)
#plot apartments in terms of first 2 components

#component 1
#high positive loadings accommodates (nr people), bedrooms, price, suggesting LISTING SIZE/VALUE (larger accommodates, more bedrooms, higher price)
#component 2
#high positive loadings reviews, overall_satisfaction, suggesting GUEST ENGAGEMENT AND SATISFACTION (more reviews, higher overall satisfaction)

import bioinfokit.visuz
bioinfokit.visuz.cluster.biplot(cscore=componentscore_abnb,
                                loadings=mypca_abnb.components_,
                                labels=abnb.columns,
                                var1=round(mypca_abnb.explained_variance_ratio_[0], 2),
                                var2=round(mypca_abnb.explained_variance_ratio_[1], 2),
                                show=True
                                )       



#component 3
#high positive loadings longitude, high negative loading on latitude, suggesting GEOGRAPHICAL LOCATION (higher longitude towards west, higher latitude towards north): so a south-west neighborhood 

#component 4
#high positive loadings on longitude and latitude, suggesting GEOGRAPHICAL LOCATION in north-west neighborhood.

bioinfokit.visuz.cluster.biplot(cscore=componentscore_abnb,
                                loadings=mypca_abnb.components_,
                                labels=abnb.columns,
                                var1=round(mypca_abnb.explained_variance_ratio_[2], 2),
                                var2=round(mypca_abnb.explained_variance_ratio_[3], 2),
                                show=True
                                )       
#component 5
#high positive loadings on reviews, low neative loading on overall_satisfaction, suggesting that some apartments have many reviews (high traffic) but low overall satisfaction (mediocre listing)
bioinfokit.visuz.cluster.biplot(cscore=componentscore_abnb,
                                loadings=mypca_abnb.components_,
                                labels=abnb.columns,
                                var1=round(mypca_abnb.explained_variance_ratio_[4], 2),
                                var2=round(mypca_abnb.explained_variance_ratio_[5], 2),
                                show=True
                                )       



#Group apartments based on the first  component
#componentscore_abnb[:, 0:1] contains the scores for the 2 components

from sklearn.cluster import KMeans

x = componentscore_abnb[:, 0:2]  # use the first two components for clustering
kmeans = KMeans(n_clusters=5, random_state=42)  # specify the number of clusters and random state for reproducibility
abnb["cluster"] = kmeans.fit_predict(x) #cluster in pca space

abnb.groupby('cluster')[['price', 'bedrooms', 'accommodates', 'reviews', "overall_satisfaction"]].mean()

#elbow method to determine optimal number of clusters
import matplotlib.pyplot as plt

inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(x)
    inertia.append(kmeans.inertia_)     


plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.xticks(range(1, 11))
plt.grid()
plt.show()  

#I can repeat this outside the pca space, based on the original features:

inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(abnb_s)#I use the original standardized features
    inertia.append(kmeans.inertia_)     


plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.xticks(range(1, 11))
plt.grid()
plt.show()  

#inertia is even higher in the original space, so PCA is useful to reduce the number of features and speed up the clustering process


