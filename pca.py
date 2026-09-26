import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
data={
    "age":[25,30,35,40,45,50],
    "income":[30000,40000,50000,60000,70000,80000],
    "Spending":[70,60,50,40,30,20],
    "savings":[1000,5000,8000,10000,15000,20000]
}
df=pd.DataFrame(data)
scaler=StandardScaler()
scaled_data=scaler.fit_transform(df)
pca=PCA(n_components=2)
pca_result=pca.fit_transform(scaled_data)

pca_df=pd.DataFrame(pca_result,columns=["PCA1","PCA2"])
explained_varience=pca.explained_variance_ratio_
print("Varience capturd by each PCA component")
print(np.round(explained_varience*100,2))#2 digit decimal
plt.figure(figsize=(8,6))
plt.scatter(pca_df["PCA1"],pca_df["PCA2"],color="black")
plt.title("pca main pattern")
plt.xlabel("PCA! main PAttern")
plt.ylabel("Pca2 minor pattern ")
plt.grid(True)
plt.show()

