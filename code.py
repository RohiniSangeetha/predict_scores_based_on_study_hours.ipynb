'''importing the packages 
os-fixing the directry
numpy for numerical operations
pandas-handling dataframes
seaborn-visualizing data'''
import os 
import numpy as np
import pandas as pd
import seaborn as sns

'''importing the train-test split from the sklearn'''
from sklearn.model_selection import train_test_split

'''fixing the directory'''
os.chdir("E:/PROGRAM&DATASET/_DATASETS")

'''creating variable for the dataframe STUDENT_SCORE'''
data1 = pd.read_csv("STUDENT_SCORE.csv")

#CHECKING THE DATATYPE OF THE VARIABLES
data1.info()

sns.set(rc={'figure.figsize':(10,10)})
sns.regplot(x=data1.Hours,y=data1.Scores)

'''importing the linear regression  from the sklearn.linear_model'''
from sklearn.linear_model import LinearRegression 

x = data1.Hours.values.reshape(-1,1)
y= data1.Scores.values.reshape(-1,1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

regressor=LinearRegression()
regressor.fit(x_train,y_train)
prediction=regressor.predict(x_test)


hours = 9.25
own_pred = regressor.predict(np.array(hours).reshape(-1,1))

print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred))

#importing the mean squared error from sklearn.metrics
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test,prediction)
print(mse)
