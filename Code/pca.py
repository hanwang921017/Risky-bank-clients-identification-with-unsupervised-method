import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def run_pca():
	# Read the file
	df=pd.read_csv('result.csv')
	# Drop the firs two coloumns
	# df = df.drop('CUST_OID', 1)
	# df = df.drop('POSTAL_CODE', 1)
	# trasform to np array
	data = df.values
	# PCA
	pca=PCA(n_components=100)
	newData=pca.fit_transform(data)
	# print(newData.shape)
	np.save("after_PCA_data.npy",newData)

	# df=pd.DataFrame(newData)
	# train = df.iloc[0:800] # first five rows of dataframe
	# test = df.iloc[800:] # first five rows of dataframe
	# np.save("after_PCA_data_train.npy",train)
	# np.save("after_PCA_data_test.npy",test)
	# print(train.shape)
if __name__ == '__main__':
	run_pca()