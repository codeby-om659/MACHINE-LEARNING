from sklearn.linear_model import LogisticRegression
X=[[1],[2],[3],[4],[5]]
y=[0,0,1,1,1]
model=LogisticRegression()
model.fit(X,y)
hours=float(input("enter how many hours you studied="))
result=model.predict([[hours]])
if result==1:
    print(f"based on your hours{hours},you are likely to pass")
else:
    print(f"based on hours{hours},you are likely to fail")
