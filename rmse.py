# Compute the Root Mean Square Error
def compute_rmse(predict, target):
    if len(target.shape) == 2:
        target = target.squeeze()
    if len(predict.shape) == 2:
        predict = predict.squeeze()
    diff = target - predict
    if len(diff.shape) == 1:
        diff = np.expand_dims(diff,axis=-1)
    rmse = np.sqrt(diff.T@diff/diff.shape[0])
    return float(rmse)