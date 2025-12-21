import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#task1

data=pd.read_csv('data/housing.csv')
plt.scatter(data['area'], data['price'])
plt.xlabel('Area')
plt.ylabel('Price')


#task2

x=data['area'].values
y=data['price'].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2)
len(x_train)

cov_xy=np.mean(x_train*y_train)-np.mean(x_train)*np.mean(y_train)
var_x=np.mean(x_train**2)-np.mean(x_train)**2

b1=cov_xy/var_x
b0=np.mean(y_train)-b1*np.mean(x_train)
print(b1, b0)

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x_train.reshape(-1, 1), y_train)
print(lin_reg.coef_, lin_reg.intercept_)


#task3

y_pred=lin_reg.predict(x_test.reshape(-1,1))
print(y_pred, y_test)

plt.scatter(x_test, y_pred)
plt.scatter(x_test, y_test)

mae=np.mean(abs(y_test-y_pred))
mse=np.mean((y_test-y_pred)**2)
R_square=lin_reg.score(x_test.reshape(-1,1), y_test)
print(mae, mse, R_square)


#task 4 and 5

plt.scatter(x, y, color='blue', label='Actual data')
all_pred_y=lin_reg.predict(x.reshape(-1,1))
plt.plot(x, all_pred_y, color='red', label='Regression line')
pred_1000=lin_reg.predict([[1000]])
plt.scatter(1000, pred_1000, color='green', s=100, label='Predicted (1000 sq ft)')

plt.xlabel('Area (sq ft)')
plt.ylabel('Price ($)')
plt.title('House Price vs Area')
plt.legend()
plt.show()
#yes, prediction for 1000 sq ft seems reasonable according to graph and actual price
# The dataset shows a strong positive linear relationship between house size and price, meaning larger houses tend to cost more. 
# The slope of the regression line represents the average price increase per square foot, 
# while the high R² score indicates that most of the price variation is explained by size alone. 
# However, this model has limitations: it only considers one feature, assumes a perfectly linear relationship, and is sensitive to outliers. 
# Additionally, the small synthetic dataset may not reflect real-world housing patterns. 
# Despite these limitations, Linear Regression provides a simple and interpretable way to predict house prices based on size.


