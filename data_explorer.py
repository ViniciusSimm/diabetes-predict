import sklearn
import pandas as pd
from sklearn import datasets
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import normaltest

X, Y = datasets.load_diabetes(return_X_y=True)
full_database = pd.DataFrame(X,columns=['age','sex','bmi','bp','s1_tc','s2_ldl','s3_hdl','s4_tch','s5_ltg','s6_glu'])
full_database['disease_progression'] = Y

print("EXPLORATORY DATA ANALYSIS\n")

print("-"*100)

print(full_database)

print("-"*100)

print("To get more information about each variable:\n")

print("Total of lines:",full_database.shape[0])
print("Unique values of sex:", len(full_database.sex.unique()))

print("-"*100)

print("Once all the other variables were quantitatives:\n")

print(full_database.drop('sex',axis=1).describe())


fig, axes = plt.subplots(1, 3)

axes[0].set_title("Age")
sns.histplot(full_database, x="age",ax=axes[0])
axes[1].set_title("bmi")
sns.histplot(full_database, x="bmi", ax=axes[1])
axes[2].set_title("bp")
sns.histplot(full_database, x="bp", ax=axes[2])

plt.show()

fig, axes = plt.subplots(2, 3)

axes[0,0].set_title("s1_tc")
sns.histplot(full_database, x="s1_tc",ax=axes[0,0])
axes[0,1].set_title("s2_ldl")
sns.histplot(full_database, x="s2_ldl", ax=axes[0,1])
axes[0,2].set_title("s3_hdl")
sns.histplot(full_database, x="s3_hdl", ax=axes[0,2])
axes[1,0].set_title("s4_tch")
sns.histplot(full_database, x="s4_tch",ax=axes[1,0])
axes[1,1].set_title("s5_ltg")
sns.histplot(full_database, x="s5_ltg", ax=axes[1,1])
axes[1,2].set_title("s6_glu")
sns.histplot(full_database, x="s6_glu", ax=axes[1,2])

plt.show()

print("-"*100)

alpha = 0.05
print("Normaltest (using an alpha of {}):".format(alpha))
for column in full_database:
    _,p_value = normaltest(full_database[column])
    H0 = p_value <= alpha
    if H0:
        print(column, "is not normal - P-value:",p_value)
    else:
        print(column, "is normal - P-value:",p_value)

print("-"*100)

print("Correlation between variables (pearson method):\n")

corr_var = full_database.corr(method ='pearson')

print(corr_var)

print("-"*100)

print("Correlation heatmap")

sns.heatmap(corr_var)
plt.show()