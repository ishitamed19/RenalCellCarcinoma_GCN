import numpy as np
import pandas as pd
import spektral
import os
import pathlib
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from scipy import sparse
from spektral.layers import GraphConv
from spektral.layers.ops import sp_matrix_to_sp_tensor
from spektral.layers import ChebConv
import gc

np.random.seed(42)
tf.compat.v1.disable_eager_execution()

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.metrics import precision_score, recall_score

metric_scores_per_pathway = []
pathways_not_used = [] #for those having <10 gene features

kegg_pathways_path = '/Users/ishitamed/Desktop/IIITH/Data/KEGG_csv'
rcc_dataset_path = '/Users/ishitamed/Downloads/GCN_Dataset/CSV'


############## LOAD INPUT DATASET ##########################
def load_dataset(path, filename, transpose=True):
	'''
		Loads the dataset and converts into its transpose with appropriate columns
	'''
	df = pd.read_csv(os.path.join(path, filename))
	df.rename(columns={"Unnamed: 0": "pid"}, inplace=True)
	if transpose:
		df = df.astype({"pid": str})
		df = df.T
		new_header = df.iloc[0] 
		df = df[1:]
		df.columns = new_header
	return df

df_kirp = load_dataset(rcc_dataset_path,'KIRP_290_tumors_log_transformed.csv',transpose=True)
df_kirc = load_dataset(rcc_dataset_path,'KIRC_518_tumors_log_transformed.csv',transpose=True)
df_kich = load_dataset(rcc_dataset_path,'KICH_81_tumors_log_transformed.csv',transpose=True)
df_kirp['y'] = 0
df_kirc['y'] = 1
df_kich['y'] = 2
data = pd.concat([df_kirp, df_kirc, df_kich])   

del df_kirp
del df_kirc
del df_kich
##########################################################

f1_weighted_per_fold = []
f1_macro_per_fold = []
f1_micro_per_fold = []
testacc_per_fold = []
precision_per_fold = []
recall_per_fold = []

f1_weighted_per_level = []
f1_macro_per_level = []
f1_micro_per_level = []
testacc_per_level = []
precision_per_level=[]
recall_per_level=[]

##########################################################

files_to_use = ['hsa04910 .csv']


for file in files_to_use:


	print(file)
	
	pathway = pd.read_csv(os.path.join(kegg_pathways_path,file))
	pathway.rename(columns={"Unnamed: 0": "idx"}, inplace=True)
	
	genes_used = set()

	for i in range(len(pathway)):
		genes_used.add(pathway.iloc[i]['from'][4:])
		genes_used.add(pathway.iloc[i]['to'][4:])

	to_remove = []
	for gene in genes_used:
		if gene not in data.columns:
			to_remove.append(gene)

	for gene in to_remove:
		genes_used.remove(gene)

	genes_used = list(genes_used)

	for gene in to_remove:
		pathway = pathway[pathway['from']!=("hsa:"+str(gene))]
		pathway = pathway[pathway['to']!=("hsa:"+str(gene))]

	nodes = len(genes_used)
	edges = len(pathway)
	print(nodes, edges)

	if(nodes<10):
		print("NOT USED: ",file)
		continue

	genes_used.sort()


	# dict to map gene_id to node_number
	node_map = {}
	count = 0
	for gene in genes_used:
		node_map[("hsa:"+str(gene))] = count
		count += 1

	# CREATE ADJACENCY MATRIX
	adjacency_matrix = np.zeros((nodes,nodes))
	for i in range(edges):
		n1 = pathway.iloc[i]['from']
		n2 = pathway.iloc[i]['to']
		n1 = node_map[n1]
		n2 = node_map[n2]
		adjacency_matrix[n1][n2] = 1

	A = sparse.csr_matrix(adjacency_matrix)

	assert adjacency_matrix.shape[0]==nodes #sanity check
	assert edges==len(pathway)

	# CREATE NODE FEATURES MATRIX
	X = data[genes_used]
	X = X.to_numpy()
	X = X.T
	assert X.shape[0]==nodes

	# CREATE TARGET LABELS
	OneHot = False
	if OneHot:
		y = []
		for i in data['y']:
			if i==0:
				y.append([1,0,0])
			elif i==1:
				y.append([0,1,0])
			elif i==2:
				y.append([0,0,1])
	else:
		y = data['y']

	y = np.asarray(y)

	# BUILDING MODEL
	# Parameters
	l2_reg = 5e-4         # Regularization rate for l2
	learning_rate = 5e-4  # Learning rate for SGD
	batch_size = 32       # Batch size
	epochs = 50         # Number of training epochs
	es_patience = 0      # Patience fot early stopping
	channels = 16           # Number of channels in the first layer
	K = 2  
	n_out = 3

	fltr = ChebConv.preprocess(A).astype('f4')
	assert fltr.shape==adjacency_matrix.shape


	f1_weighted_per_fold.clear()
	f1_macro_per_fold.clear()
	f1_micro_per_fold.clear()
	testacc_per_fold.clear()
	precision_per_fold.clear()
	recall_per_fold.clear()

	f1_weighted_per_level.clear()
	f1_macro_per_level.clear()
	f1_micro_per_level.clear()
	testacc_per_level.clear()
	precision_per_level.clear()
	recall_per_level.clear()

	kfold1 = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
	kfold2 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

	fold = 1

	for train_idx_1, test_idx_1 in kfold2.split(X.T, y):
		X_train, X_test = X.T[train_idx_1], X.T[test_idx_1]
		X_train, X_test = X_train[..., None], X_test[..., None]
		y_train, y_test = y[train_idx_1], y[test_idx_1]
		f1_weighted_per_fold.clear()
		f1_macro_per_fold.clear()
		f1_micro_per_fold.clear()
		testacc_per_fold.clear()
		precision_per_fold.clear()
		recall_per_fold.clear()
		for train_ix, test_ix in kfold1.split(X_train, y_train):
			train_X, test_X = X_train[train_ix], X_train[test_ix]
			train_y, test_y = y_train[train_ix], y_train[test_ix]

			N = X_train.shape[-2]      # Number of nodes in the graphs
			F = X_train.shape[-1]      # Node features dimensionality

			# Model definition
			X_in = Input(shape=(N, F))
			A_in = Input(tensor=sp_matrix_to_sp_tensor(fltr))

			# dropout_1 = Dropout(dropout)(X_in)
			bn_1 = BatchNormalization()(X_in)
			graph_conv_1 = ChebConv(32,
									K=K,
									activation='relu',
									kernel_regularizer=l2(l2_reg),
									use_bias=False)([bn_1, A_in])
			# dropout_2 = Dropout(dropout)(graph_conv_1)
			bn_2 = BatchNormalization()(graph_conv_1)
			graph_conv_2 = ChebConv(32,
									K=K,
									activation='relu',
									use_bias=False)([bn_2, A_in])
			flatten = Flatten()(graph_conv_2)
			fc_1 = Dense(64, activation='relu')(flatten)
			dropout_1 = Dropout(0.3, seed=42)(fc_1)
			fc_2 = Dense(32, activation='relu')(dropout_1)
			output = Dense(n_out, activation='softmax')(fc_2)

			# Build model
			model = Model(inputs=[X_in, A_in], outputs=output)
			optimizer = Adam(lr=learning_rate)
			model.compile(optimizer=optimizer,
						  loss='sparse_categorical_crossentropy',
						  metrics=['acc'])


			# Train model
			validation_data = (test_X, test_y)
			model.fit(train_X,
					  train_y,
					  batch_size=16,
					  validation_data=validation_data,
					  epochs=10, verbose=0)

			y_pred = model.predict(X_test, verbose=1)
			y_p = []
			for row in y_pred:
				y_p.append(np.argmax(row))
			target_names = ['0', '1', '2']
			print("Fold: ", fold)
			fold += 1
			# print(classification_report(y_test, y_p, target_names=target_names))
			f1_weighted_per_fold.append(f1_score(y_test, y_p, average='weighted'))
			f1_macro_per_fold.append(f1_score(y_test, y_p, average='macro'))
			f1_micro_per_fold.append(f1_score(y_test, y_p, average='micro'))
			testacc_per_fold.append(accuracy_score(y_test, y_p))
			precision_per_fold.append(precision_score(y_test, y_p,  average='micro'))
			recall_per_fold.append(recall_score(y_test, y_p,  average='micro'))

			if fold<30:
				del model
				del train_y
				del test_y
				del train_X
				del test_X
				gc.collect()
	 

		f1_weighted_per_level.append(np.mean(f1_weighted_per_fold))
		f1_macro_per_level.append(np.mean(f1_macro_per_fold))
		f1_micro_per_level.append(np.mean(f1_micro_per_fold))
		testacc_per_level.append(np.mean(testacc_per_fold))
		precision_per_level.append(np.mean(precision_per_fold))
		recall_per_level.append(np.mean(recall_per_fold))

	# APPEND METRICS
	scores = [file, np.mean(f1_weighted_per_level), np.mean(f1_macro_per_level), np.mean(f1_micro_per_level), np.mean(testacc_per_level), np.mean(precision_per_level), np.mean(recall_per_level)]
	print(scores)
	# metric_scores_per_pathway.append(scores)

	# GENERATE OUTPUT CSV
	full_data = X.T
	full_data = full_data[..., None]
	gcn_pathway_output = model.predict(full_data)
	filename_output_csv = os.path.join("/Users/ishitamed/Desktop/IIITH/Data/GCN_pathway_output_csv",file)
	np.savetxt(filename_output_csv,gcn_pathway_output)
	# np.savetxt(os.path.join("/content/drive/My Drive/IIITH/GCN_KEGG/GCN_pathway_output_scores",file),scores)

	# REMOVE GARBAGE
	del pathway
	del X_train
	del X_test
	del train_X
	del test_X
	del A
	del adjacency_matrix
	del X
	del y
	del train_y
	del test_y
	del y_train
	del y_test
	del fltr
	del genes_used
	del node_map
	del to_remove
	del model
	gc.collect()
# SAVE METRICS FOR ALL PATHWAYS
# metric_scores_df = pd.DataFrame(metric_scores_per_pathway, index=["Name", "f1-weighted", "f1-macro", "f1-micro", "test-acc", "prec", "recall"])
# metric_scores_df.to_csv(index=False)