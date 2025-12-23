import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

#task1
df=pd.read_csv('data/housing.csv')

df.head()
df.dtypes

yes_no_cols=['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
df[yes_no_cols]=df[yes_no_cols].apply(lambda x: x.map({'yes': 1, 'no': 0}))
df.head()

df=pd.get_dummies(df, columns=['furnishingstatus'], dtype=int)
df.head()

num_cols=['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
scaler=StandardScaler()
df[num_cols]=scaler.fit_transform(df[num_cols])
df.head()


#task2
x=df.drop('price', axis=1)
y=df['price']

x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.2, random_state=42)

trainer=LinearRegression()
trainer.fit(x_train, y_train)

trainer.coef_
feature_coefficients = pd.DataFrame({
    'Feature': x_train.columns,
    'Coefficient': trainer.coef_
})

print(feature_coefficients)
trainer.intercept_

#task3
y_pred=trainer.predict(x_test)
print(y_pred,y_test)

mae=np.mean(abs(y_test-y_pred))
mse=np.mean((y_test-y_pred)**2)
R_squared=trainer.score(x_test, y_test)
print(mae,mse,R_squared)



#task4
import matplotlib.pyplot as plt
import seaborn as sns

corr=df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()

y_all_pred=trainer.predict(x)
plt.figure(figsize=(8,6))
plt.scatter(y, y_all_pred, alpha=0.6, color='blue') 
plt.plot([y.min(), y.max()], [y_all_pred.min(), y_all_pred.max()], color='red', linewidth=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Predicted vs Actual House Prices')
plt.show()


#task5
new_house = pd.DataFrame({
    'area': [2400],
    'bedrooms': [4],
    'bathrooms': [3],
    'stories': [2],
    'mainroad': [1],         
    'guestroom': [0],      
    'basement': [1],         
    'hotwaterheating': [0],  
    'airconditioning': [1],
    'parking': [2],
    'prefarea': [1],         
    # One-hot encoded furnishingstatus (assuming 'semi-furnished' only)
    'furnishingstatus_furnished': [0],
    'furnishingstatus_semi-furnished': [1],
    'furnishingstatus_unfurnished': [0]
})


num_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
new_house[num_cols] = scaler.transform(new_house[num_cols])  # trained scaler


predicted_price = trainer.predict(new_house)
print("Predicted Price:", predicted_price[0])

# The model provides a reasonable first approximation of house prices using the given features. 
# It highlights the importance of size and key amenities but is constrained by linearity assumptions, 
# limited feature set, and potential multicollinearity. For higher accuracy, more complex models 
# (e.g., Random Forests or Gradient Boosting) or additional data on location and market trends could be explored.