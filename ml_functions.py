#This program will combine any number of ML outputs to create a super output
#as training input, there needs to be a list of star names attached to the label, the features, and the binned lightcurve/periodogram data.  Most of the machine learning methods take the numerical features, but the CNN needs the binned lc/pdgrm.
#It is assumed here that all of the input data is in the correct format, but has not yet been scaled.

seed=1212

#This function will split up the data into a training and testing set. It returns a list of the object names in each category that you can then use to split up the data appropriately. 
def get_training():
	return (train_names, test_names)


def optimize_RF(X,Y):
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.model_selection import GridSearchCV
	ntrees = [50,100,150,200,250,500]
	nbranches = [3,6,9,12]
	nfeatures = [2,4,6,8,10,12]
	param_grid = {'n_estimators':ntrees,'max_features':nfeatures,'max_depth':nbranches}
	grid_search = GridSearchCV(RandomForestClassifier(), param_grid,cv=3)
	grid_search.fit(X,Y)
	print(grid_search.best_params_)
	return (grid_search.best_params_['n_estimators'], grid_search.best_params_['max_features'],grid_search.best_params_['max_depth'])


#This function runs a random forest algorithm using the training dataset.
#Assumes that you have already found the best-fit tuning parameters
def run_RF(X,Y, test_X, ntrees=200, nbranches=6, nfeatures=6, optimize=False):
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.externals import joblib
	import os
	save_file = 'final_RF_model.sav'
	if os.path.isfile(save_file):
		single_forest = joblib.load(save_file)
	else:
		if optimize:
			print('optimizing parameters for the Random Forest...')
			ntrees, nfeatures, nbranches = optimize_RF(X,Y)
		single_forest = RandomForestClassifier(n_estimators = ntrees, max_features=nfeatures, max_depth=nbranches, random_state=seed, class_weight='balanced_subsample') 
		single_forest.fit(X, Y)
		joblib.dump(single_forest, save_file)
	RF_preds = single_forest.predict(test_X)

	return RF_preds

def optimize_KNN(X,Y):
	import numpy as np
	from sklearn.model_selection import GridSearchCV
	from sklearn.neighbors import KNeighborsClassifier
	n_neighbors = [7,8,9,10,11,12,13,14,15]
	weights = ['uniform','distance']
	param_grid = {'n_neighbors':n_neighbors, 'weights':weights}
	grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, scoring='f1_weighted',cv=3) #class_weight='balanced', dual=False
	grid_search.fit(X, Y)
	print(grid_search.best_params_)
	return (grid_search.best_params_['n_neighbors'], grid_search.best_params_['weights'])

#This function runs the K-nearest-neighbor algorithm 
def run_KNN(X,Y,test_X, nneighbors=10, weights='distance', optimize=False):
	from sklearn.neighbors import KNeighborsClassifier
	from sklearn.externals import joblib
	import os
	save_file = 'final_KNN_model.sav'
	if os.path.isfile(save_file):
		KNN_clf = joblib.load(save_file)
	else:
		if optimize:
			print('optimizing parameters for the KNN...')
			nneighbors, weights = optimize_KNN(X,Y)
		KNN_clf = KNeighborsClassifier(n_neighbors=nneighbors, weights=weights)
		KNN_clf.fit(X,Y)
		joblib.dump(KNN_clf, save_file)
	KNN_preds = KNN_clf.predict(test_X)

	return KNN_preds


def optimize_SVC(X,Y):
	from sklearn.svm import SVC
	from sklearn.model_selection import GridSearchCV
	Cs = [0.001, 0.01, 0.1, 1, 10, 50, 100, 1000]
	tol = [0.00005, 0.0001,0.0005, 0.001, 0.005, .01, .05]
	kernel = ['linear', 'poly','rbf']
	param_grid = {'C': Cs, 'tol' : tol, 'kernel':kernel}
	grid_search = GridSearchCV(SVC(class_weight='balanced', random_state=seed, gamma='scale'), param_grid, cv=3) #class_weight='balanced', dual=False
	grid_search.fit(X, Y)
	print(grid_search.best_params_)
	return(grid_search.best_params_['C'], grid_search.best_params_['tol'], grid_search.best_params_['kernel'])



#This function runs the linear SVC
def run_SVC(X,Y,test_X, c=80, tol=0.0003, optimize=False):
	from sklearn.svm import SVC
	from sklearn.externals import joblib
	import os
	save_file = 'final_SVC_model.sav'
	if os.path.isfile(save_file):
		SVC_clf = joblib.load(save_file)
	else:
		if optimize:
			print('optimizing parameters for the SVC...')
			c, tol, kernel = optimize_SVC(X,Y)
		SVC_clf = SVC(class_weight='balanced', random_state=seed, C=c, tol=tol, kernel=kernel, gamma='scale') #assumed you have optimized C and tol from, ie GridSearchCV 
		SVC_clf.fit(X, Y) 
		joblib.dump(SVC_clf, save_file)
	SVC_preds = SVC_clf.predict(test_X)
	return SVC_preds


def optimize_LR(X,Y):
	from sklearn.linear_model import LogisticRegression
	from sklearn.model_selection import GridSearchCV
	Cs = [10, 20, 30, 40, 50, 60,70, 80, 90, 100]
	tol = [0.0001,0.0002, 0.0003, 0.0004, .0005, .0006, .0007, 0.0008]
	param_grid = {'C': Cs, 'tol' : tol}
	grid_search = GridSearchCV(LogisticRegression(class_weight='balanced', random_state=12), param_grid, scoring='f1_weighted', cv=3) #class_weight='balanced', dual=False
	grid_search.fit(X, Y)
	print(grid_search.best_params_)
	return(grid_search.best_params_['C'], grid_search.best_params_['tol'])

#This function does a logistic regression
def run_LR(X,Y,test_X, c=90, tol=0.005, optimize=False):
	from sklearn.linear_model import LogisticRegression
	from sklearn.externals import joblib
	import os
	save_file = 'final_LR_model.sav'
	if os.path.isfile(save_file):
		LR_clf = joblib.load(save_file)
	else:
		if optimize:
			print('optimizing parameters for the logistic regression...')
			c, tol = optimize_LR(X,Y)
		print('C: %s, tol: %s'%(c, tol))
		LR_clf = LogisticRegression(random_state=seed, class_weight='balanced', C=c, tol=tol, multi_class='multinomial', solver='lbfgs')
		LR_clf.fit(X,Y)
		joblib.dump(LR_clf, save_file)
	LR_preds = LR_clf.predict(test_X)
	return LR_preds



#This defines the form of the CNN
def baseline_model(f1=8,l1=4,f2=10,l2=8):
	print(f1,f2,l1,l2)
	from keras.optimizers import Adamax
	from keras.models import Sequential
	from keras.layers import Dense, Dropout, Activation, Flatten
	from keras.layers.convolutional import Conv1D, MaxPooling1D
	model = Sequential()
	model.add(Conv1D(f1, l1, input_shape=(500,2), padding='causal', activation='relu'))
	print('###########')
	print(model.layers[-1].output_shape)
	model.add(MaxPooling1D(pool_size=2))
	model.add(Dropout(0.4))
    
	#model.add(Conv1D(8, 4, padding='causal', activation='relu'))
	#model.add(MaxPooling1D(pool_size=2))
	#print(model.layers[-1].output_shape)
	#model.add(Dropout(0.3))
    
	model.add(Conv1D(f2, l2, padding='causal', activation='relu'))
	model.add(MaxPooling1D(pool_size=2))
	print(model.layers[-1].output_shape)
	model.add(Dropout(0.4))
    
	model.add(Flatten()) 
	print(model.layers[-1].output_shape)
  

	model.add(Dense(512, activation='relu', kernel_initializer='he_normal')) 
	model.add(Dropout(0.4))
	model.add(Dense(1024, activation='relu', kernel_initializer='he_normal')) 
	model.add(Dropout(0.4))
	model.add(Dense(4, activation='sigmoid', kernel_initializer='he_normal')) 
	print('###########')
	print(model.layers[-1].output_shape)
	optimizer = Adamax(lr=0.001)
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model

def optimize_CNN(X,Y):
	from keras.wrappers.scikit_learn import KerasClassifier
	from sklearn.model_selection import RandomizedSearchCV
	param_grid = dict(f1=[4,6,8,10], l1=[4,6,8,10], f2=[6,8,10,12], l2=[4,6,8,10], epochs = [50,100], batch_size=[10,20,30])
	grid = RandomizedSearchCV(KerasClassifier(build_fn=baseline_model), param_grid, n_jobs=1, n_iter=2)
	grid_result = grid.fit(X,Y, shuffle=True)
	print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
	means = grid_result.cv_results_['mean_test_score']
	stds = grid_result.cv_results_['std_test_score']
	params = grid_result.cv_results_['params']
	for mean, stdev, param in zip(means, stds, params):
	    print("%f (%f) with: %r" % (mean, stdev, param))
	return (grid_result.best_params_['f1'],grid_result.best_params_['f2'], grid_result.best_params_['l1'], grid_result.best_params_['l2'], grid_result.best_params_['epochs'],grid_result.best_params_['batch_size'])
   
###############################################################################
#this function runs the convolutional neural network
def run_CNN(X, Y, test_X, epochs=175, batch_size=20,f1=8,l1=4,f2=10,l2=8,optimize=False):
	from keras.wrappers.scikit_learn import KerasClassifier
	from sklearn.externals import joblib
	import os
	save_file = 'final_CNN_model.sav'
	if os.path.isfile(save_file):
		CNN_clf = joblib.load(save_file)
	else:
		if optimize:
			print("finding optimal parameters...")
			f1, f2, l1, l2, epochs, batch_size = optimize_CNN(X, Y)
		CNN_clf = KerasClassifier(build_fn=baseline_model,f1=f1,l1=l1,f2=f2,l2=l2)
		CNN_clf.fit(X,Y,epochs=epochs, batch_size=batch_size)
		joblib.dump(CNN_clf, save_file)
	CNN_preds = CNN_clf.predict(test_X)
	return CNN_preds