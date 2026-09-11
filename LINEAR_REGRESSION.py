from sklearn.linear_model import LinearRegression
X=[[1],[2],[3],[4],[5]]
y=[40,50,65,75,90]
model=LinearRegression()
model.fit(X,y)
hours=float(input("enter how many hours you studiedd="))
pridected_marks=model.predict([[hours]])
print(f"based on your {hours}you may score around{pridected_marks}")
