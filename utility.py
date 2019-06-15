import numpy as np
from sklearn.metrics import mean_squared_error

def rmsle(labels,predictions,exponential_transformation=True):
    if exponential_transformation:
        labels = np.exp(labels)
        predictions = np.exp(predictions)
    root_mean_squared_error = np.mean((np.log(labels)-np.log(predictions))**2)**0.5

    return root_mean_squared_error

def simple_rmsle(labels,predictions):
    error = np.sqrt(mean_squared_error(labels,predictions))

    return error

def h1_correlation(df,target,subject,permutations=1*10**4):
    testing_df = df.loc[:,[target,subject]].dropna()
    target_array = np.array(testing_df[target].dropna())
    subject_array = np.array(testing_df[subject].dropna())

    base_correlation = np.corrcoef(target_array,subject_array)[0,1]
    test_correlations = []

    for i in range(permutations):
        target_test = np.random.permutation(target_array)
        subject_test = np.random.permutation(subject_array)

        test_correlation = np.corrcoef(target_test,subject_test)[0,1]
        test_correlations.append(test_correlation)

    p_value = np.mean(np.array(([abs(value)>abs(base_correlation) for value in test_correlations])))

    return base_correlation, p_value
