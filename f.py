import requests
import turtle 
from tkinter import Tk, Label, Button  
from sklearn import datasets  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import keras 
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd 
import numpy as np

# Function to construct customer URL
def get_customer_url(customer_id):
    return f"http://ec2-3-111-8-121.ap-south-1.compute.amazonaws.com/api/get-customer?id={customer_id}"


for i in range(1, 5000):
    url = get_customer_url(i)
    response = requests.get(url=url)

    
    if response.status_code == 200:
        print(url)

turtle.forward(100)
turtle.right(90)
turtle.forward(100)
turtle.done()

root = Tk()
label = Label(root, text="Hello, GUI!")
button = Button(root, text="Click me")
label.pack()
button.pack()
root.mainloop()

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)


model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=8))
model.add(Dense(units=1, activation='sigmoid'))

data = {'Name': ['John', 'Jane', 'Bob'], 'Age': [28, 35, 22]}
df = pd.DataFrame(data)

array = np.array([[1, 2, 3], [4, 5, 6]])
sum_result = np.sum(array)

print("Sum of the array:", sum_result)
