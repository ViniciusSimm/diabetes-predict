from opcode import hasconst
import numpy as np
from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import statsmodels.api as sm
from sklearn.svm import SVC, SVR

def score_model(model,X_test,Y_test):
    Y_predict = model.predict(X_test)
    R2_score = metrics.r2_score(Y_test,Y_predict)
    print("R2_score:",R2_score)
    MSE = metrics.mean_squared_error(Y_test,Y_predict)
    MSE_sqrt = np.sqrt(MSE)
    print("MSE:",MSE_sqrt)
    return R2_score

def train_model(model,X_train,Y_train):
    model.fit(X_train,Y_train)
    return model


X, Y = datasets.load_diabetes(return_X_y=True)
full_database = pd.DataFrame(X,columns=['age','sex','bmi','bp','s1_tc','s2_ldl','s3_hdl','s4_tch','s5_ltg','s6_glu'])
full_database['disease_progression'] = Y

x = full_database.drop('disease_progression',axis=1)
y = full_database['disease_progression']

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.3,random_state=100)

print("Train size:", X_train.shape[0])
print("Test size:", X_test.shape[0])
print("Total:", X.shape[0])

print("-"*100)

print("\nLinear Regression:")

model_LR = LinearRegression()
model_LR = train_model(model_LR,X_train,Y_train)
score_LR = score_model(model_LR,X_test,Y_test)

print("\nLinear Regression without s2 and s4:")

model_LR_2 = LinearRegression()
model_LR_2 = train_model(model_LR_2,X_train.drop(['s2_ldl','s4_tch'],axis=1),Y_train)
score_LR_2 = score_model(model_LR_2,X_test.drop(['s2_ldl','s4_tch'],axis=1),Y_test)

print("\nOrdinary Least Squares (statsmodels):")

X_train_plus_constant = sm.add_constant(X_train)
model_OLS = sm.OLS(Y_train,X_train_plus_constant,hasconst=True).fit()
print(model_OLS.summary())

print("\nSVC model:")

model_SVC = SVC()
model_SVC = train_model(model_SVC,X_train,Y_train)
score_SVC = score_model(model_SVC,X_test,Y_test)

print("\nSVC model without s2 and s4:")

model_SVC_2 = SVC()
model_SVC_2 = train_model(model_SVC_2,X_train.drop(['s2_ldl','s4_tch'],axis=1),Y_train)
score_SVC_2 = score_model(model_SVC_2,X_test.drop(['s2_ldl','s4_tch'],axis=1),Y_test)

print("\nSVR model:")

model_SVR = SVR()
model_SVR = train_model(model_SVR,X_train,Y_train)
score_SVR = score_model(model_SVR,X_test,Y_test)