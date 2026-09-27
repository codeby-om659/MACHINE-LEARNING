import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#create a dataset
data={
    'year_experience':[1.1,1.3,1.5,2.0,2.2,2.9,3.0,3.2,3.2,3.7,4.0,4.5,4.9,5.1],
    'salary':[39343,46205,37731,43525,39891,56642,60150,54445,64445,57189,63218,61111,67938,66029]
}
df=pd.DataFrame(data)

X=df[['year_experience']]
y=df['salary']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(X_train,y_train)

new_experience=pd.DataFrame({
    'year_experience':[3.5]
})
prediction=model.predict(new_experience)
print(prediction[0])
