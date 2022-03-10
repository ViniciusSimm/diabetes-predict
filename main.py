import sklearn
import pandas as pd
from sklearn import datasets
import seaborn as sns
from matplotlib import pyplot as plt

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


fig, axes = plt.subplots(1, 3)

axes[0].set_title("Age")
sns.histplot(full_database, x="age",ax=axes[0])
axes[1].set_title("bmi")
sns.histplot(full_database, x="bmi", ax=axes[1])
axes[2].set_title("bp")
sns.histplot(full_database, x="bp", ax=axes[2])

plt.show()