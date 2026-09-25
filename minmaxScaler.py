import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
data={
    'StudyHours':[1,2,3,4,5],
    'Testscore':[40,50,60,70,80]
}
df=pd.DataFrame(data)
minmax_scaler=MinMaxScaler()
minmax_scaled=minmax_scaler.fit_transform(df)
print("minmax scaler output")
print(pd.DataFrame(minmax_scaled,columns=['StudyHours','Testscore'])),

# TrainTesttSplit
X=df[['StudyHours']]
y=df[['Testscore']]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=42)
print("Trainng data")
print(X_train)
print("testing data")
print(X_test)
print("trainig data")
print(y_train)
print("testing data")
print(y_test)
