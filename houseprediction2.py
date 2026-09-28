import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score

# step 1 Synthesis Dataset
np.random.seed(42)
n_sample =500

size_sqft=np.random.randint(500,3500,size=n_sample)
bedrooms=np.random.randint(1,6,size=n_sample)
age_years=np.random.randint(1,30,size=n_sample)
Distance_city_km=np.random.uniform(1.0,25.0,size=n_sample)

#price formuka
price=(
    size_sqft*3500
    +bedrooms*150000
    -age_years*20000
    -Distance_city_km*50000
    +np.random.normal(0,100000,size=n_sample)
)

df= pd.DataFrame({
    'size_sqft':size_sqft,
    'brdrooms':bedrooms,
    'age_year':age_years,
    'Distance_city_km':Distance_city_km,
    'price':price
})
print(df.head())
# step 2  X and y
X=df.drop(columns=['price'])
y=df['price']

# train dat set
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)

#feturing Scaling
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.fit_transform(X_test)

#step 4
model=RandomForestRegressor(n_estimators=100,random_state=42)
model.fit(X_train_scaled,y_train)

#step 5 model evaluation
y_pred=model.predict(X_test_scaled)
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
r2=r2_score(y_test,y_pred)

print("model performance")
print(f'root mean squared:{rmse}')
print(f'r2 score is :{r2}')

#step 6
def predict_house(size,beds,age,distance):
    sample_data=pd.DataFrame([[size,beds,age,distance]],columns=X.columns)
    scaled_sample=scaler.transform(sample_data)
    prediction_value=model.predict(scaled_sample)[0]
    return prediction_value
#test with a custom house 
sample_price=predict_house(1200,2,5,8.5)
print(f'prdicted price for custom house: rs{sample_price:,.2f}')
