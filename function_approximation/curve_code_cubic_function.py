# Imports
import matplotlib.pyplot as plt
import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense

from tensorflow.python.framework import test_util
from tensorflow.python.client import device_lib

# Printing device properties
print(device_lib.list_local_devices())
print(test_util.IsMklEnabled())

# Load training data
x = np.random.uniform(-3, 7, (10000, 1))
y = x*x*x - 6*x*x + 4*x + 12

# # Define model
# model = Sequential()
# model.add(Dense(140, input_dim=1, activation='relu'))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')
# prefit = time.time()
# model.fit(x, y, epochs=50, batch_size=50)
# postfit = time.time()

# # Define model
model = Sequential()
model.add(Dense(80, input_dim=1, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
prefit = time.time()
model.fit(x, y, epochs=50, batch_size=50)
postfit = time.time()


print("test data:")
test_data = np.asarray([-2, -1, 0, 1, 2, 3, 4, 5, 6])
print(test_data)
print("-----------")

print("expected output:")
squarer = lambda t: t*t*t - 6*t*t + 4*t + 12
expceted_outcome = np.array([squarer(xi) for xi in test_data])
print(expceted_outcome)
print("-----------")

print("predictions:")
predictions = model.predict(test_data)
print(predictions)
print("-----------")

c = 0
percentage_off = 0
# this while asumes that the prediction and expected outcome both are negative at the same time
while c < predictions.size:
    temp_var = 0
    if abs(predictions[c]) > abs(expceted_outcome[c]):
        temp_var = (abs(predictions[c]) - abs(expceted_outcome[c])) / abs(predictions[c])
    else:
        temp_var = (abs(expceted_outcome[c]) - abs(predictions[c])) / abs(expceted_outcome[c])
    c = c + 1
    percentage_off = percentage_off + temp_var

print("Fitting duration: {0} seconds".format((postfit-prefit)))
print("Off by atleast: {0}%".format((percentage_off / predictions.size) * 100))

ynew = model.predict(x)

plt.subplot(2, 1, 1)
plt.scatter(x, y, color='blue', s=1)
plt.title('y = $cubic function$')
plt.ylabel('Real y')

plt.subplot(2, 1, 2)
plt.scatter(x, ynew, color='red', s=1)
plt.xlabel('x')
plt.ylabel('Approximated y')

plt.show()
