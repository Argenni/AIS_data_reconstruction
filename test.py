import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


series = np.array([[1, 2],
           [1.2, -1],
           [-1.5, -1],
           [-2, 2]], dtype=float)
model = sm.tsa.VAR(series)
model = model.fit(maxlags=3, trend='n')

"""
series = np.array([-1.5, -1.2, -1, 1.01])
model = sm.tsa.ARIMA(endog=series, order=(3,1,3)).fit()
"""
pred = model.forecast(series, steps=1)
print(pred)

plt.figure()
plt.scatter(range(series.shape[0]), series[:,0], color='b')
plt.scatter(series.shape[0], pred[0][0], color='r')
plt.scatter(range(series.shape[0]), series[:,1], color='k')
plt.scatter(series.shape[0], pred[0][1], color='g')
plt.show(block=True)