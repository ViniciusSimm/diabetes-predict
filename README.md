# diabetes-predict


----DATA EXPLORER----


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

Analysing the correlation between every variable using pandas' method corr(), it's possible to see a very strong corelation between s1_tc and s2_ldl, and an inverse strong relation between s3_hdl and s4_tch, and somehow strong between s2_ldl and s4_tch. The correlation can also be seen on a heatmap.

Usually, when two variables are strongly correlated, one of them is excluded to make the get a simpler model.

We also did some exploring with scipy normaltest. Adopting an alpha of 0.05, we tested all the variables to see if they came from a normal distribution. The results indicate that only s6_glu actually came from a normal distribution (accepting the Null Hyphotesis). All the other variables, although it seemed like normal distributions, they were not.

So, after the data exploring, we were ready to start modeling.


----MODELING----


The first model was Linear Regression.

The data was subdivided in train and test and the Linear Regression model was trained and evaluated. The first way to measure the score was by using R_2 (coefficient of determination). But one of the downsides of this score model is that R_2 does not decrease by the increase of distinct variables.
A second way to compare models is the mean squared error (MSE). This parameter is affected by the number of variables in the model.

Our goal is to INCREASE R_2 and DECREASE MSE.

Worth saying that Linear Regression is one of the most simple and easy correlations to be done, where we try to find a straight line that summarize the data.
The results were not great, so we tried other models.

The OLS regression had a better R_2 then the linear model used before.

We tried a non-linear model to predict the results. Scikit provides us with an alternative using Support Vector Machines: SVC.

As we can see by the results, the non-linear model had a significant disadvantage on both R2_score and MSE. A SVR model was also tested but the result was equally underperformed.

Now we were able to use DummyRegressor, from Scikit, to evaluate how good our model was so far. It's results where worst than every other model tested before, what means our model is somehow good and applicable.

But we had to keep trying new models and, after finding one that pleased us enought, start working with parameters to improve it even more.

The next model was Decision Tree Regressor, but it's performance was even worse than the Dummy.
The Random Forest Regressor had a better outcome than the Decision Tree Regressor once it tested many trees and chose the one with the best outcome.


----IMPROVING BEST MODELS----


With six distinct models, we chose the top three to improve by changing its hyperparameters.
The models were: Linear Regression, OLS and Random Forest Regressor.

For the Linear Regression we have used RFE (that eliminates some variables up to a determined number).

Although the OLS model had a slightly better outcome than the Linear Regression one, both models didn't have many hyperparameters to be tested.

The only model that actually was improved by iterating hyperparameters was Random Forest Regressor. The RandomizedSearchCV was used to test the combination between many hyperparameters available.

It worth saying that the outcome was improved by changing the proportion of data used for training and for testing.
We had to decrease significantly the amount of records used to test the model, that is explained by the amount of data given at the database (the first thing we have explored).
Problably the results would be even better if we had a larger dataset to work with.

A final table was created to compare the results between all selected models.

We will use Linear Regression to formulate our final model, because the results were better than the other models and it's easier to work with than OLS.