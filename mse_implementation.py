import numpy as np
def mse(yhat, y_true):
    sum = 0.0
    for i in range(len(yhat)):
        sum += np.square(y_true[i]-yhat[i])
    return sum/len(yhat)
def rmse(predictions, targets):
    differences = predictions - targets
    differences_squared = differences ** 2
    mean_of_differences_squared = differences_squared.mean()
    rmse_val = np.sqrt(mean_of_differences_squared)
    return rmse_val
yhat = np.array([0.000, 0.166, 0.333])
y_true = np.array([0.000, 0.254, 0.998])
print('d value is ' + str(['%.8f' % elem for elem in yhat]))
print("p value is " + str(["%.8f" % elem for elem in y_true]))
print(f' the mse value is {mse(yhat, y_true)}')
rmse_val = rmse(yhat, y_true)
print("rms error is: " + str(rmse_val))
