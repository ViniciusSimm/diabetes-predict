import sklearn
import pandas as pd
from sklearn import datasets

X, Y = datasets.load_diabetes(return_X_y=True)
full_database = pd.DataFrame(X,columns=['age','sex','bmi','bp','s1_tc','s2_ldl','s3_hdl','s4_tch','s5_ltg','s6_glu'])
full_database['disease_progression'] = Y

print(full_database)

print("-"*100)

print("To get more information about each variable:\n")

print("Total of lines:",full_database.shape[0])
print("Unique values of sex:", len(full_database.sex.unique()))

print("-"*100)

print("Once all the other variables were quantitatives:")

print(full_database.drop('sex',axis=1).describe())