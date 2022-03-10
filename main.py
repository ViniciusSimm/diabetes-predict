import sklearn
import pandas as pd
from sklearn import datasets

diabetes_X, diabetes_Y = datasets.load_diabetes(return_X_y=True)
full_database = pd.DataFrame(diabetes_X,columns=['age','sex','bmi','bp','s1_tc','s2_ldl','s3_hdl','s4_tch','s5_ltg','s6_glu'])
full_database['disease_progression'] = diabetes_Y