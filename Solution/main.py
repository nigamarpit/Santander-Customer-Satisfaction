import gc
import xgboost as xgb
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

if __name__ == "__main__":
		
		train = pd.read_csv('trainPreProcess.csv')
		test = pd.read_csv('testPreProcess.csv')
		features = train.columns[1:-1]
		train.insert(1, 'SumZeros', (train[features] == 0).astype(int).sum(axis=1))
		test.insert(1, 'SumZeros', (test[features] == 0).astype(int).sum(axis=1))

		remove = []
		c = train.columns
		for i in range(len(c)-1):
				v = train[c[i]].values
				for j in range(i+1, len(c)):
						if np.array_equal(v, train[c[j]].values):
								remove.append(c[j])

		train.drop(remove, axis=1, inplace=True)
		test.drop(remove, axis=1, inplace=True)

		remove = []
		for col in train.columns:
				if train[col].std() == 0:
						remove.append(col)

		train.drop(remove, axis=1, inplace=True)
		test.drop(remove, axis=1, inplace=True)
		features = train.columns[1:-1]
		pca = PCA(n_components=2)
		x_train_projected = pca.fit_transform(normalize(train[features], axis=0))
		x_test_projected = pca.transform(normalize(test[features], axis=0))
		train.insert(1, 'PCAOne', x_train_projected[:, 0])
		train.insert(1, 'PCATwo', x_train_projected[:, 1])
		test.insert(1, 'PCAOne', x_test_projected[:, 0])
		test.insert(1, 'PCATwo', x_test_projected[:, 1])
		tokeep = ['num_var39_0', 
							'ind_var13',  
							'num_op_var41_comer_ult3',  
							'num_var43_recib_ult1',  
							'imp_op_var41_comer_ult3',  
							'num_var8',  
							'num_var42',  
							'num_var30',  
							'saldo_var8',  
							'num_op_var39_efect_ult3',  
							'num_op_var39_comer_ult3', 
							'num_var41_0',  
							'num_op_var39_ult3',  
							'saldo_var13',  
							'num_var30_0',  
							'ind_var37_cte', 
							'ind_var39_0', 
							'num_var5',  
							'ind_var10_ult1',  
							'num_op_var39_hace2', 
							'num_var22_hace2', 
							'num_var35', 
							'ind_var30', 
							'num_med_var22_ult3',  
							'imp_op_var41_efect_ult1', 
							'var36', 
							'num_med_var45_ult3', 
							'imp_op_var39_ult1',  
							'imp_op_var39_comer_ult3', 
							'imp_trans_var37_ult1', 
							'num_var5_0',  
							'num_var45_ult1',  
							'ind_var41_0', 
							'imp_op_var41_ult1',  
							'num_var8_0', 
							'imp_op_var41_efect_ult3', 
							'num_op_var41_ult3', 
							'num_var22_hace3',
							'num_var4', 
							'imp_op_var39_comer_ult1', 
							'num_var45_ult3', 
							'ind_var5',  
							'imp_op_var39_efect_ult3',  
							'num_meses_var5_ult3', 
							'saldo_var42', 
							'imp_op_var39_efect_ult1', 
							'PCATwo',  
							'num_var45_hace2',  
							'num_var22_ult1',  
							'saldo_medio_var5_ult1',  
							'PCAOne', 
							'saldo_var5', 
							'ind_var8_0',  
							'ind_var5_0',  
							'num_meses_var39_vig_ult3', 
							'saldo_medio_var5_ult3', 
							'num_var45_hace3', 
							'num_var22_ult3', 
							'saldo_medio_var5_hace3',  
							'saldo_medio_var5_hace2', 
							'SumZeros',  
							'saldo_var30',  
							'var38',  
							'var15'] 
		features = train.columns[1:-1]
		todrop = list(set(tokeep).difference(set(features)))
		train.drop(todrop, inplace=True, axis=1)
		test.drop(todrop, inplace=True, axis=1)
		features = train.columns[1:-1]
		split = 10
		skf = StratifiedKFold(train.TARGET.values,
													n_folds=split,
													shuffle=False,
													random_state=42)

		train_preds = None
		test_preds = None
		visibletrain = blindtrain = train
		index = 0

		num_rounds = 440
		params = {}
		params["objective"] = "binary:logistic"
		params["eta"] = 0.025
		params["subsample"] = 0.95
		params["colsample_bytree"] = 0.65
		params["silent"] = 1
		params["max_depth"] = 5
		params["min_child_weight"] = 0.9
		params["eval_metric"] = "auc"
		params["seed"] = 1
		
		for train_index, test_index in skf:
				print('Fold:', index)
				visibletrain = train.iloc[train_index]
				blindtrain = train.iloc[test_index]
				dvisibletrain = \
						xgb.DMatrix(csr_matrix(visibletrain[features]),
												visibletrain.TARGET.values,
												silent=True)
				dblindtrain = \
						xgb.DMatrix(csr_matrix(blindtrain[features]),
												blindtrain.TARGET.values,
												silent=True)
				watchlist = [(dblindtrain, 'eval'), (dvisibletrain, 'train')]
				clf = xgb.train(params, dvisibletrain, num_rounds,
												evals=watchlist, early_stopping_rounds=50,
												verbose_eval=False)

				blind_preds = clf.predict(dblindtrain)
				print('Blind Log Loss:', log_loss(blindtrain.TARGET.values,
																					blind_preds))
				print('Blind ROC:', roc_auc_score(blindtrain.TARGET.values,
																					blind_preds))
				index = index+1
				del visibletrain
				del blindtrain
				del dvisibletrain
				del dblindtrain
				gc.collect()
				dfulltrain = \
						xgb.DMatrix(csr_matrix(train[features]),
												train.TARGET.values,
												silent=True)
				dfulltest = \
						xgb.DMatrix(csr_matrix(test[features]),
												silent=True)
				if(train_preds is None):
						train_preds = clf.predict(dfulltrain)
						test_preds = clf.predict(dfulltest)
				else:
						train_preds *= clf.predict(dfulltrain)
						test_preds *= clf.predict(dfulltest)
				del dfulltrain
				del dfulltest
				del clf
				gc.collect()

		train_preds = np.power(train_preds, 1./index)
		test_preds = np.power(test_preds, 1./index)
		print('Average Log Loss:', log_loss(train.TARGET.values, train_preds))
		print('Average ROC:', roc_auc_score(train.TARGET.values, train_preds))
		submission = pd.DataFrame({"ID": train.ID,
															 "TARGET": train.TARGET,
															 "PREDICTION": train_preds})

		submission.to_csv("simplexgbtrain111.csv", index=False)
		submission = pd.DataFrame({"ID": test.ID, "TARGET": test_preds})
		submission.to_csv("simplexgbtest111.csv", index=False)

