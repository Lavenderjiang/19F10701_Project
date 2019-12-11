import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from HDF5Dataset import HDF5Dataset
from math import sqrt
import matplotlib

# Create the dataset
rng = np.random.RandomState(1)

filename = '/home/erynqian/10701/19F10701_Project/testData/sampled/first365.hdf5'
ds = HDF5Dataset(filename)
X, y, valX, valY = ds.train_val_test()

# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=5)

regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=5),
                          n_estimators=300, random_state=rng, learning_rate=0.5)

regr_1.fit(X, y)
regr_2.fit(X, y)

# Predict
y_1 = regr_1.predict(valX)
y_2 = regr_2.predict(valX)

# Plot the results
X = np.arange(len(valX))
plt.figure()
# plt.scatter(X, y, c="k", label="training samples")
# plt.plot(X, y_1, c="g", label="n_estimators=1", linewidth=2)
# plt.plot(X, y_2, c="r", label="n_estimators=300", linewidth=2)
# plt.xlabel("data")
# plt.ylabel("target")
# plt.ylim((0,600))
# plt.title("Boosted Decision Tree Regression")
# plt.legend()
# plt.savefig("adaboost.png")

plt.scatter(valY[:1000], y_2[:1000])
plt.xlabel("truth")
plt.ylabel("predict")
plt.plot(list(range(80)))
plt.title("Boosted Decision Tree Regression")
plt.savefig("adaboost_predicts.png")

# RMSE
err = 0
for i,j in zip(y, y_2):
    err += (i-j)**2
err = err / len(y)
err = sqrt(err)
print("RMSE:", err)

# Absolute error
from sklearn.metrics import mean_absolute_error
print("mean_absolute_error: ", mean_absolute_error(valY, y_2))
