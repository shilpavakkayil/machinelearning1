import numpy as np
y_hat = np.array([0.000, 0.166, 0.333])
y_true = np.array([0.000, 0.254, 0.998])
print("d is: " + str(["%.8f" % elem for elem in y_hat]))
print("p is: " + str(["%.8f" % elem for elem in y_true]))
def mae(predictions, targets):
    differences = predictions - targets
    print(differences)
    absolute_differences = np.absolute(differences)
    print(absolute_differences)
    mean_absolute_differences = absolute_differences.mean()
    return mean_absolute_differences
mae_val = mae(y_hat, y_true)
print ("mae error is: " + str(mae_val))