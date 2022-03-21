import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

df=pd.read_excel('Dataframe.xlsx')

def adjusted_r2(score,k,n):
    adj_r2=1-(((1-score)*(n-1))/(n-k-1))
    return adj_r2

# X,y=df[["Average Discount_x1", "other_discount%_x2", "magicpin_discount%_x3",
#        "phonepe_discount%_x4", "store_cred%_x5", "Listed Discount%_x6",
#        "Google Spends_x7", "Affiliates Spends_x8", "Facebook Spends_x9"]],df[["Amount_Paid_Y"]]


X,y=df[["other_discount%_x2", "phonepe_discount%_x4", "store_cred%_x5", "Listed Discount%_x6",
       "Google Spends_x7", "Affiliates Spends_x8", "Facebook Spends_x9"]],df[["Amount_Paid_Y"]]

n=len(y)
k=len(X.columns)

poly= PolynomialFeatures(degree=1, include_bias=False)
poly_features=poly.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.3, random_state=1)

poly_reg_model = LinearRegression()
poly_reg_model.fit(X_train,y_train)

poly_reg_y_predicted=poly_reg_model.predict(X_test)
poly_reg_rmse=np.sqrt(mean_squared_error(y_test,poly_reg_y_predicted))
score=r2_score(y_test,poly_reg_y_predicted)
adj_r2=adjusted_r2(score, k, n)
print("R-squared: "+str(score))
print("Adjusted R-squared: "+str(adj_r2))

# lin_reg_model = LinearRegression()
# lin_reg_model.fit(X,y)
