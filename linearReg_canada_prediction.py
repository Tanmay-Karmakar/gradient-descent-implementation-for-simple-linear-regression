
# Import the required libraries
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 

# Read the data
data = pd.read_csv("canada_per_capita_income.csv")

# Define the soruse and the target variable and the total number of sample points m
fr = data["year"]
y = data["per capita income (US$)"]
fr = np.array(fr)
yr = np.array(y)
m = len(fr)

# Scale the featers
maxf = 0
for i in fr:         # find the maximum value of the features
    if maxf < i:
        maxf = i
x = [i/maxf for i in fr]
x = np.array(x)

# Initialize the linear regression parameter w, b
w = 10000
b = 0

# Define the model f
def f(w,b,x):
    return w*x + b

# Define the cost function J
def J(w,b,x):
    y_hat = f(w,b,x)
    z = y_hat - y
    return (1/(2*len(x)))*np.dot(z,z)

# Define the derivative of the cost function w.r.t w and b
def J_w(w,b,x):
    y_hat = f(w,b,x)
    z = y_hat - y
    return (1/len(x))*np.dot(z,x)
def J_b(w,b,x):
    y_hat = f(w,b,x)-y
    return (1/len(x))*np.sum(y_hat)

# Define the learning rate a
a = 1

# Stopping constant e
e = 10

# Iterations
iterations = 0
prev = 0
while(abs(prev - J(w,b,x))>=e):
    prev = J(w,b,x)
    w,b = (w-a*J_w(w,b,x)),(b - a*J_b(w,b,x))
    iterations +=1

print(f"w = {w}, b = {b}, Iterations = {iterations}")

print(f"cost = {J(w,b,x)}")

# Find the actual values of w and b without scaling
w = w/maxf
b = b

# Plot the regression line
g = [1970,1981,2016]
h = [f(w,b,i) for i in g]
plt.plot(g,h)
plt.scatter(fr,y)
plt.show()

# Predict per capita income
y = 'yes'
while(y == ('yes' or 'Yes' or 'YES')):
    year = int(input("Enter the year: "))
    income = f(w,b,year)
    print(f"The predicted income per capita in {year} is {income}")
    y = input("Do you want to predict again? Type yes or no.\n")

# Compare with sklearn
    

'''from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(data[['year']],y)

print(model.coef_, w,sep=",")
print(model.intercept_, b,sep=",")'''