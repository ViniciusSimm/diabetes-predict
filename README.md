# diabetes-predict

For this project, we used the diabetes dataset from sklearn.
The data was already mean centered and scaled, but the first thing to be done was to understand the meaning of every column and the data description.

All the information was given on scikit learn webpage:

age
sex
bmi (body mass index)
bp (average blood pressure)
s1_tc (total serum cholesterol)
s2_ldl (low-density lipoproteins)
s3_hdl (high-density lipoproteins)
s4_tch (total cholesterol / HDL)
s5_ltg (possibly log of serum triglycerides level)
s6_glu (blood sugar level)

And the output (Y) was described as:
"A quantitative measure of disease progression one year after baseline"

After checking the unique values for each variable, "sex" was the only qualitative. Therefore, it was the only variable not considered for the next data evaluation.

After analysing each graphic, all of them seem to be close to a normal graph, except "s4_tch", that shows an unusual shape.

Analysing the correlation between every variable using pandas' method corr(), it's possible to see a very strong relation between s1_tc and s2_ldl, and an inverse strong relation between s3_hdl and s4_tch, and somehow strong between s2_ldl and s4_tch. The correlation can also be seen on a heatmap.

Usually, when two variables are strongly correlated, one of them is excluded to make the get a simpler model.

We also did some exploring with scipy normaltest. Adopting an alpha of 0.05, we tested all the variables to see if they came from a normal distribution. The results indicate that only s6_glu actually came from a normal distribution (accepting the Null Hyphotesis). All the other variables, although it seemed like normal distributions, they were not.

So, after the data exploring, we were ready to start modeling.