import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
data={
    "std_hours":[1,2,2.5,3,4,5,6,7,8,9],
    "attendence":[40,50,55,58,65,70,75,80,85,90],
    "passed":[0,0,0,0,1,1,1,1,1,1]
}

df=pd.DataFrame(data)
X=df[['std_hours','attendence']]
y=df[['passed']]
# split karege
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#model train
model=LogisticRegression()
model.fit(X_train,y_train)
prediction=model.predict(X_test)
new_std=pd.DataFrame({
    'std_hours':[3.5,1],
    'attendence':[62,30]
})
prediction=model.predict(new_std)
print(prediction)

for i in range(0,2):
    if prediction[i]==1:
        print("pass")
    else:
        print("failed")
