import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

# This function calculates the Root Mean Square Error (RMSE) between the true and predicted values.

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# This function generates a plot comparing the actual and predicted values.
# It also prints the RMSE value for the predictions.

def results(y_true, y_pred, title):
    error = rmse(y_true, y_pred)
    print(f"RMSE: {error:.2f} MW")
    plt.figure(figsize=(12, 6))
    plt.plot(y_true.index, y_true, label='Actual', color='blue')
    plt.plot(y_true.index, y_pred, label='Predicted', color='red')
    plt.title(title)
    plt.xlabel("Date")
    plt.xticks(rotation=45)
    plt.ylabel("Megawatts")
    plt.legend()
    plt.show()