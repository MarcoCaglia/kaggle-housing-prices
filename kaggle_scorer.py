import numpy as np

def rmsle(labels,predictions):
    root_mean_squared_error = np.mean((np.log(labels)-np.log(predictions))**2)**0.5

    return root_mean_squared_error
