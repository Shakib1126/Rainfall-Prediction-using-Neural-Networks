import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor

dataset =  pd.read_csv("austin_final.csv")
print(dataset)

X = dataset.drop(['PrecipitationSumInches'], axis = 1) 
  
# the output or the label. 
Y = dataset['PrecipitationSumInches'] 
# reshaping it into a 2-D vector 
Y = Y.values.reshape(-1, 1) 

sc= MinMaxScaler()
X= sc.fit_transform(X)
Y=sc.fit_transform(Y)

x1= sc.inverse_transform(X)
print(x1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

def build_regressor():
    regressor = Sequential()
    regressor.add(Dense(units=17, input_dim=17))
    regressor.add(Dense(units=1))
    regressor.compile(optimizer='adam', loss='mean_squared_error',  metrics=['mae','accuracy'])
    return regressor

regressor = KerasRegressor(build_fn=build_regressor, batch_size=32,epochs=15)
results=regressor.fit(X_train,y_train)
y_pred= regressor.predict(X_test)

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

inp = np.array([[74], [60], [45], [67], [49], [43], [33], [45], 
                [57], [29.68], [10], [7], [2], [0], [20], [4], [31]]) 
inp = inp.reshape(1, -1)
#inp = sc.fit_transform(inp)
output = regressor.predict(inp)
#output = sc.inverse_transform(output)
print("The precipitation in inches is ",abs(output))
print("The precipitation in Millimeters is ",abs(output*25.4))
  
