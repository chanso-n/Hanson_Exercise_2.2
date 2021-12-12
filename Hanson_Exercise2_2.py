# import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# import data
df_car = pd.read_csv('car_data.csv')

# assign data
x_given = np.array(df_car['weight'])
y_given = np.array(df_car['hwy_mpg'])
x = df_car['weight'].values[:,np.newaxis]
y = df_car['hwy_mpg'].values

# Q1. make scatter plot
plt.scatter(x_given, y_given, s = 20)

# label axes
plt.xlabel('x values')
plt.ylabel('y values')

# set x and y limits
plt.xlim(0, 5000)
plt.ylim(0, 60)

# Q2. It looks like it is a negative slope. Heavier vehicles tend to get less hwy mpg
# Q3. I would expect the slope to be negative. These two variables seem to be negatively related
# Q4. This would be a negative slope. Meaning -
# you would move +1 up the y axis and -20 (left) on the x axis -
# or -1 down the y axis and +20 (right) on the x axis. Every 20 lbs of vehicle weight decreases -
# the hwy mpg by 1


# Q5. add a line to plot
# set m (slope) and b (y-intercept) values
m = -0.013
b = 65

# create x values for line
x_values = np.linspace(1400, 4100, 205)
y_values = m*x_values + b

# send new line to plot
plt.plot(x_values, y_values, c = 'k')

# Q5. Answer: slope = -0.013, y-intercept = 65


# Q6. best-fit line
# use sklearn to fit linear model
model = LinearRegression()
model.fit(x, y)

# send best-fit line to plot
plt.plot(x, model.predict(x), color = 'r')


# get prediction and best-fit values
my_line = np.array(y_values)
pred_values = np.array(model.predict(x))


# Q7. RMSE
# find RMSE of my prediction and the actual data
Pred_MSE = np.square(np.subtract(y_given, my_line)).mean()
Pred_RMSE = np.sqrt(Pred_MSE)
print(Pred_RMSE)
# Q7. My predicted line RMSE: 12.437605035136105

# find RMSE of best-fit line and the actual data
Best_MSE = np.square(np.subtract(y_given, pred_values)).mean()
Best_RMSE = np.sqrt(Best_MSE)
print(Best_RMSE)
# Q7. Best-fit line RMSE: 4.144895442072008

# Q7. Answer: The best-fit line has a lower RMSE, meaning its points are situated more closely -
# to the actual data points than my prediction line. Therefore, the best-fit line is a better -
# predictor of hwy mpg as a function of weight than my prediction line. 


# Q8. predict mpg of a car that weighs 3200 lbs
print(model.predict([[3200]]))
# Q8. Answer: mpg would be 23.95427129 for a 3200 lbs car


# show plot
plt.show()



                       
