# ML-workflow
Introduce the ML workflow: data loading, exploration, preprocessing, training, evaluation, and reporting. You will train a linear regression model on the California Housing dataset

Dataset 
Use the California Housing dataset (built into scikit-learn or available on Kaggle).

Steps by step
1. create a virtualenv and install pandas, numpy, scikit-learn , matplotlib, seaborn , juptyer.
2. Load dateset (sklearn.datsets.fetch_california_housing) or Kaggle CSV.
3. perform EDA:Check distribution  , correction , corrrelation , missing values.
4. select features, split data(train_test_spilt).
5. train LinearRegression ; evaluate using MAE , RMSE ,R^2.
6. Plot predicated vs actual Scatter, residuals.

CODE:-


#basics
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np


data=fetch_california_housing(as_frame=True)
df=pd.concat([data.data, data.target.rename('MedHouseVal')],axis=1)
df.head()

#train/test
X=df.drop(columns='MedHouseVal')
Y=df['MedHouseVal']
X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size=0.2,random_state=42)


#model
model=LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

#metrics
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, Y_pred)
print(f"MAE:{mae:.3f} RMSE:{rmse:.3f} R2:{r2:.3f}")

plt.scatter(Y_test,Y_pred, alpha=0.4)
plt.xlabel("Actual")
plt.ylabel("Predicated")
plt.title("Actual vs Predicated")
plt.plot([min(Y_test), max(Y_test)], [min(y_test), max(y_test)], color='red')
plt.show()

