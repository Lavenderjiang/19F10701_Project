import xgboost as xgb
import h5py
import numpy as np
import HDF5Dataset
import matplotlib.pyplot as plt

trainDataset = "/home/erynqian/10701/19F10701_Project/testData/sampled/first365.hdf5"
evalDataset = "/home/erynqian/10701/19F10701_Project/testData/sampled/first50.hdf5"

files = []

# Load training dataset
train_dset = HDF5Dataset.HDF5Dataset(trainDataset)
train_data, train_label = train_dset()
print(train_data.shape, train_label.shape)

# Load validation dataset
eval_dset = HDF5Dataset.HDF5Dataset(evalDataset)
eval_data, eval_label = eval_dset()
print(eval_data.shape, eval_label.shape)

# Initialize training inputs
dtrain = xgb.DMatrix(train_data, label=train_label)
dtest = xgb.DMatrix(eval_data, label=eval_label)
param = {'max_depth': 12, 'eta': 0.5, 
        'subsample': 0.9, 'booster' : 'gbtree',
        'lambda': 1., 'colsample_bytree': 0.9, 'early_stopping_rounds': 5,
        'objective': 'reg:squarederror', 'nthread': 4, 'eval_metric' : 'rmse'}
evallist = [(dtest, 'eval'), (dtrain, 'train')]

# Training
num_round = 50
result = {}
bst = xgb.train(param, dtrain, num_round, evallist, evals_result=result)
print(result)
bst.save_model('0001.model')
bst.dump_model('dump.raw.txt')

# Continue training the existing model
model_path = "/home/erynqian/10701/19F10701_Project/0001.model"
# bst = xgb.train(param, dtrain, 1, evallist, xgb_model=model_path)
for trainDataset in files:
    # Load training dataset
    hdf5_filename = trainDataset.split('.')[0] + '.hdf5'
    train_dset = HDF5Dataset.HDF5Dataset(hdf5_filename)
    train_data, train_label = train_dset()
    dtrain = xgb.DMatrix(train_data, label=train_label)
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    num_round = 100
    print("Start training on", trainDataset)
    bst = xgb.train(param, dtrain, num_round, evallist, xgb_model=model_path)

# Plot loss
eval_loss, train_loss = result['eval']['rmse'], result['train']['rmse']
it = list(range(len(eval_loss)))
# plt.plot(it, eval_loss)
plt.plot(it, train_loss)
plt.xlabel('Iterations')
plt.ylabel('RMSE loss')
plt.title('XGBoost loss over time')
plt.savefig('XGBoost_loss.png')

# Test
plt.figure()
ypred = bst.predict(dtrain, ntree_limit=bst.best_ntree_limit)
ytruth = train_dset.Y()
plt.scatter(ytruth[:10000], ypred[:10000])
plt.xlabel('Truth')
plt.ylabel('Predictions')
r = list(range(300))
plt.plot(r)
plt.xlim((0,300))
plt.ylim((0,300))
plt.title('XGBoost')
plt.savefig('XGBoost.png')


# Plotting
xgb.plot_importance(bst)
# xgb.plot_tree(bst, num_trees=2)