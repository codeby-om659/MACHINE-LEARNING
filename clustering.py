import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data={
    "customer":["om","aman","ansh","prem","rahul","rajesh"],
    "age":[20,30,40,22,38,25],
    "spending":[100,200,300,110,290,130]
}
df=pd.DataFrame(data)
X=df[["age","spending"]]
model=KMeans(n_clusters=3,random_state=42,n_init=10)
df["group"]=model.fit_predict(X)
plt.figure(figsize=(8,7))
for group in df["group"].unique():
    group_data=df[df["group"]==group]
    plt.scatter(group_data["age"],group_data["spending"],label=f"group{group}")
plt .xlabel("age")
plt.ylabel("spending_score")
plt.title("customer Segment(k-Means)")
plt.legend()
plt.grid(True)
print(df)
plt.show()
