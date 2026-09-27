import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data={
    'Bedrooms':[1,2,2,3,3,4,4],
    'sqft':[500,800,1000,1200,15000,1800,2000],
    'Location':['city','suburbs','city','suburbs','city','city','suburbs'],
    'price_lakh':[25,40,52,55,75,90,105]
}

df=pd.DataFrame(data)
label=LabelEncoder()
df['location_Encoder']=label.fit_transform(df['Location'])
print(df)
X=df[['Bedrooms','sqft','location_Encoder']]
y=df['price_lakh']

model=LinearRegression()
model.fit(X,y)

new_data=pd.DataFrame({
    'Bedrooms':[3],
    'sqft':[1300],
    "location_Encoder":[0]
})
prediction=model.predict(new_data)
print(prediction[0])
