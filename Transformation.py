import math
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np

data = pd.read_csv('Mewing data set.csv')
data.fillna(0, inplace=True)
# print(data.head())

x = data['Comp Time']
y = data.salary

plt.scatter(x,y)
plt.xlabel("Comp Time")
plt.ylabel("Salary")
plt.title("Scatter plot of Comp Time and Salary")
m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b, color="red")
plt.show()

#Adjusted r-squared .489

# x = sm.add_constant(x)
# model = sm.OLS(y, x).fit()

# print(model.summary())