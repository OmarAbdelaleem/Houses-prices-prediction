#!/usr/bin/env python
# coding: utf-8

# # Houses' prices predictions
# by/Omar Abdelaleem

# # Getting and setting up the data

# In[2]:


import numpy as np
import pandas as pd
from pandas import Series,DataFrame

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


from sklearn.datasets import load_boston

boston = load_boston()

print (boston.DESCR)


# # Doing a quick visualization fo the data we have.

# In[4]:


plt.hist(boston.target,bins=50)

plt.xlabel('Prices in $1000s')
plt.ylabel('Number of houses')


# Interesting, now let's see a scatter plot of one feature, versus the target. In this case we'll use the housing price versus the number of rooms in the dwelling.

# In[5]:


plt.scatter(boston.data[:,5],boston.target,label=True)

plt.xlabel=('Number of rooms')
plt.ylabel=('Price in $1000s')


# #### Great! Now we can make out a slight trend that price increases along with the number of rooms in that house, which intuitively makes sense!

# In[6]:


boston_df = DataFrame(boston.data)

boston_df.columns = boston.feature_names

boston_df.head()


# In[7]:


boston_df['Price'] = boston.target


# In[8]:


boston_df.head()


# # The mathematics behind the Least Squares Method:
# 
# In this particular lecture we'll use the least squares method as the way to estimate the coefficients. Here's a quick breakdown of how this method works mathematically

# ## Now, we're labeling each line as having a distance D, and each point as having a coordinate of (X,Y). Then we can define our best fit line as the line having the property were:
# 
# D21+D22+D23+D24+....+D2N
# 
# So how do we find this line? The least-square line approximating the set of points:
# 
# (X,Y)1,(X,Y)2,(X,Y)3,(X,Y)4,(X,Y)5,
# 
# has the equation:
# 
# Y=a0+a1X
# 
# this is basically just a rewritten form of the standard equation 
# for a line:
# 
# Y=mx+b
# 
# We can solve for these constants a0 and a1 by simultaneously 
# 
# solving these equations:
# 
# ΣY=a0N+a1ΣX
# 
# ΣXY=a0ΣX+a1ΣX2
# 
# These are called the normal equations for the least squares line. There are further steps that can be taken in rearranging these equations to solve for y, but we'll let scikit-learn do the rest of the heavy lifting here.

# In[10]:


X = boston_df.RM

#The next line is for adding ones to the arrays of the attributes to be like: (X,1)
X=np.vstack([boston_df.RM,np.ones(len(boston_df.RM))]).T

#Y= Target price of the houses
Y = boston_df.Price

#Using Numpy to creat the single variable linear refression
#Creat an array of [X 1]

#Now get out m and b values for our best fit line
#linalg = linear Algebra, lstsq=Least squares
m,b = np.linalg.lstsq(X,Y)[0]

#The[0] is because we only want the first value of that index

#b= The slope,  M= The intercept 
print(m, b)

# First the original points, Price vs Avg Number of Rooms
plt.plot(boston_df.RM,boston_df.Price,'x')


# Next the best fit line
x= boston_df.RM

plt.plot(x, m*x + b,'r',label='Best Fit Line')


# # Getting the error
# 
# We've just completed a single variable regression using the least squares method! We see that the resulting array has the total squared error. For each element, it checks the the difference between the line and the true value (our original D value), squares it, and returns the sum of all these. This was the summed D^2 value we discussed earlier.
# 
# It's probably easier to understand the root mean squared error, which is similar to the standard deviation. In this case, to find the root mean square error we divide by the number of elements and then take the square root.
# 
# 

# In[11]:


# Get the resulting array
result = np.linalg.lstsq(X,Y)

# Get the total error
error_total = result[1]

# Get the root mean square error
rmse = np.sqrt(error_total/len(X) )

# Print
print("The root mean squared error was %.2f " %rmse)


# Since the root mean square error (RMSE) corresponds approximately to the standard deviation we can now say that the price of a house won't vary more than 2 times the RMSE 95% of the time.
# 
# Thus we can reasonably expect a house price to be within $13,200 of our line fit.

# In[12]:


import sklearn
from sklearn.linear_model import LinearRegression


# In[13]:


lreg = LinearRegression()


# In[14]:


# Data Columns
X_multi = boston_df.drop('Price',1)

# Targets
Y_target = boston_df.Price


# In[15]:


lreg.fit(X_multi,Y_target)


# In[16]:


print("The estimated intercepts coefficient is %.2f"%lreg.intercept_)

print("The number of coefficients used was %d"%len(lreg.coef_))


# So we have basically made an equation for a line, but instead of just one coefficient m and an intercept b, we now have 13 coefficients.

# In[18]:


coeff_df = DataFrame(boston_df.columns)
coeff_df.columns = ['Features']

coeff_df['Coefficient estimate'] = Series(lreg.coef_)

coeff_df


# Just like we initially plotted out, it seems the highest correlation between a feature and a house price was the number of rooms.

# # Using Training and Validation
# 

# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


# Grab the output and set as X and Y test and train data sets!
X_train,X_test,Y_train,Y_test = train_test_split(X,boston_df.Price)


# In[22]:


# Print shapes of the training and testing data sets
print(X_train.shape,X_test.shape,Y_test.shape,Y_train.shape)


# # Predicting Prices

# In[23]:


#Creating new linear regression object
lreg = LinearRegression()

#fitting the train datasets for that object
lreg.fit(X_train,Y_train)


# In[24]:


pred_train = lreg.predict(X_train)
pred_test = lreg.predict(X_test)


# In[25]:


#Now we will get the mean square error

print("Fit a model X_train, and calculate MSE with Y_train: %.2f"  % np.mean((Y_train - pred_train) ** 2))
    
print("Fit a model X_train, and calculate MSE with X_test and Y_test: %.2f"  %np.mean((Y_test - pred_test) ** 2))


# It looks like our mean square error between our training and testing was pretty close. But how do we actually visualize this?

# # Residual Plots
# In regression analysis, the difference between the observed value of the dependent variable (y) and the predicted value (ŷ) is called the residual (e). Each data point has one residual, so that:
# 
# Residual=Observedvalue−Predictedvalue
# 
# You can think of these residuals in the same way as the D value we discussed earlier, in this case however, there were multiple data points considered.

# A residual plot is a graph that shows the residuals on the vertical axis and the independent variable on the horizontal axis. If the points in a residual plot are randomly dispersed around the horizontal axis, a linear regression model is appropriate for the data; otherwise, a non-linear model is more appropriate.
# 
# 

# In[27]:


train = plt.scatter(pred_train,(pred_train - Y_train),c='b',alpha=0.5)

test = plt.scatter(pred_test,(pred_test - Y_test),c='r',alpha=0.5)

plt.hlines(y=0,xmin=-10,xmax=60)

plt.legend((train,test),('Training','Test'),loc='lower left')

plt.title('Residual Plots')


# Looks like there aren't any major patterns to be concerned about, it may be interesting to check out the line occuring towards the bottom right, but overall the majority of the residuals seem to be randomly allocated above and below the horizontal.

# In[28]:


# Residual plot of all the dataset using seaborn
sns.residplot('RM', 'Price', data = boston_df)

