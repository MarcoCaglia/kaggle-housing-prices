import numpy as np

def rmsle(labels,predictions,exponential_transformation=False):
    if exponential_transformation:
        labels = np.exp(labels)
        predictions = np.exp(predictions)
    root_mean_squared_error = np.mean((np.log(labels)-np.log(predictions))**2)**0.5

    return root_mean_squared_error

def rmsle_validation(ytest, ypred):
	assert len(ytest) == len(ypred)
	return np.sqrt(np.mean((np.log1p(ypred) - np.log1p(ytest))**2))
