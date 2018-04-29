from sklearn.linear_model import Ridge
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import linear_model
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

train_data = pd.read_csv('reg_train.csv')
test_data = pd.read_csv('reg_test.csv')

#convert date into numerical value
train_data['dteday'] = pd.to_datetime(train_data['dteday'])
train_data['dteday']=train_data['dteday'].map(dt.datetime.toordinal)
test_data['dteday'] = pd.to_datetime(test_data['dteday'])
test_data['dteday']=test_data['dteday'].map(dt.datetime.toordinal)


X_features =['instant','dteday','season','yr','mnth','hr','holiday','weekday','workingday','weathersit','temp','atemp','hum','windspeed','casual','registered']
y_features = ['cnt']

# split into input (X) and output (Y) variables
#X_train = train_data[:,0:18]
#y_train = train_data[:,18]
X_train = train_data[X_features].values
y_train = train_data[y_features].values
X_test = train_data[X_features].values
y_test= train_data[y_features].values


"""#reg = linear_model.LinearRegression()
#cv_results = cross_val_score(reg, X_train, y_train, cv=5)
#print(cv_results)
#reg.fit(X_train, y_train) #train model on train data
#reg.score(X_train, y_train) #check score
ridge = Ridge(alpha=0.1, normalize=True)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
ridge.score(X_test, y_test)
"""

# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(16, input_dim=16, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model


# define the model
def larger_model():
	# create model
	model = Sequential()
	model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
	model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)


kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, X_train, y_train, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))


# evaluate model with standardized dataset
numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=larger_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X_train, y_train, cv=kfold)
print("Larger: %.2f (%.2f) MSE" % (results.mean(), results.std()))