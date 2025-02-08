"""
Group:       Marc Semadeni and Sarah Martin
Assignment:  Regression Transformations
Date:        February 7, 2025
"""

import math
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np

# --------------------------------
# Step 1: Least Squares Regression
# --------------------------------

# Import data; replace nulls
data = pd.read_csv('Mewing data set.csv')
data.fillna(0, inplace=True)

X = data['Comp Time']
y = data.salary
x=X
plt.scatter(x,y)
plt.xlabel("Comp Time")
plt.ylabel("Salary")
plt.title("Scatter plot of Comp Time and Salary")
sns.regplot(x=x,y=y, line_kws={"color":"red"})
plt.show()

# Adjusted r-squared .489

x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
print(model.summary())

# ------------------------------
# Step 2: Linear Transformations
# ------------------------------


# ---- Logarithmic Transformation -----
log_x=np.log1p(X)
plt.scatter(log_x,y)
plt.xlabel("Comp Time")
plt.ylabel("Salary")
plt.title("Log scatter plot of Comp Time and Salary")
sns.regplot(x=log_x,y=y, line_kws={"color":"red"})
plt.show()

# Adjusted r-squared: .199

log_x = sm.add_constant(log_x)
model = sm.OLS(y, log_x).fit()
print(model.summary())


# ---- Exponential Transformation -----
x_squared=X**2
plt.scatter(x_squared,y)
plt.xlabel("Comp Time")
plt.ylabel("Salary")
plt.title("Exponential scatter plot of Comp Time and Salary")
sns.regplot(x=x_squared,y=y, line_kws={"color":"red"})
plt.show()

# Adjusted r-squared: 0.644

x_squared = sm.add_constant(x_squared)
model = sm.OLS(y, x_squared).fit()
print(model.summary())

# ---- Power Transformation -----
power_of_x=1.05**X
plt.scatter(power_of_x,y)
plt.xlabel("Comp Time")
plt.ylabel("Salary")
plt.title("Power scatter plot of Comp Time and Salary")
sns.regplot(x=power_of_x,y=y, line_kws={"color":"red"})
plt.show()

# Adjusted r-squared: 0.669

power_of_x = sm.add_constant(power_of_x)
model = sm.OLS(y, power_of_x).fit()
print(model.summary())

# ------------------------------
# Step 3: General Inverse
# ------------------------------

pow_of_y=1.05**y
plt.scatter(X,pow_of_y)
plt.xlabel("Comp Time")
plt.ylabel("Salary")
plt.title("General Inverse scatter plot of Comp Time and Salary")
sns.regplot(x=X,y=pow_of_y, line_kws={"color":"red"})
plt.show()

# Adjusted r-squared .732

x = sm.add_constant(X)
model = sm.OLS(pow_of_y, X).fit()
print(model.summary())