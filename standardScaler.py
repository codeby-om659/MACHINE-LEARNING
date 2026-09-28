import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
data={
    'StudyHours':[1,2,3,4,5],
    'Testscore':[40,50,60,70,80]
}
df=pd.DataFrame(data)
Std_scaler=StandardScaler()
std_scaled=Std_scaler.fit_transform(df)
print("Standard sclaer output")
print(pd.DataFrame(std_scaled,columns=['studyHours','Testscore']))
