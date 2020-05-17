# 5b) I find it harder to adjust the hyperparameters for predicting sin.
# I have noticed it is harder to get a good model at 10 compared to 0 when using data within that interval.
# However it looks like the model gets more precise when expaning the traning data to 15 or 20

# Imports
import matplotlib.pyplot as plt
import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense

from tensorflow.python.framework import test_util
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print(test_util.IsMklEnabled())

# Load training data
x = np.random.random((10000, 1)) * 10
y = np.sin(x)

# Define model
model = Sequential()
model.add(Dense(40, input_dim=1, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
prefit = time.time()
model.fit(x, y, epochs=100, batch_size=50)
postfit = time.time()


print("test data:")
test_data = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(test_data)
print("-----------")

print("expected output:")
expceted_outcome = np.sin(test_data)
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
    # print(temp_var)
    c = c + 1
    percentage_off = percentage_off + temp_var

print("Fitting duration: {0} seconds".format((postfit-prefit)))
print("Off by atleast: {0}%".format((percentage_off / predictions.size) * 100))

ynew = model.predict(x)

plt.subplot(2, 1, 1)
plt.scatter(x, y, color='blue', s=1)
plt.title('y = $sin(x)$')
plt.ylabel('Real y')

plt.subplot(2, 1, 2)
plt.scatter(x, ynew, color='red', s=1)
plt.xlabel('x')
plt.ylabel('Approximated y')

plt.show()
