import numpy as np
import pandas as pd
import os
import operator
from collections import Counter 
from sklearn.model_selection import train_test_split


base_path = 'data'

def read_data(base_path, filename):
	csv_file = os.path.join(base_path, filename)
	df = pd.read_csv(csv_file)
	X = df.iloc[:, :-1]
	y = df.iloc[:, -1]
	return X, y

def fit(X, y):
	global X_global
	global y_global
	X_global = X
	y_global = y

def distance_between(x, X_test):
	return np.linalg.norm(x - X_test)

def predict(X_test, k):
	global X_global
	X = X_global
	global y_global
	y = y_global
	processed_observations = []
	for feature, label in zip(X, y):
		distance = distance_between(feature, X_test)
		processed_observations.append((label, distance))
	processed_observations = sorted(processed_observations, key=operator.itemgetter(1))[1:k+1]
	counter = Counter([item[0] for item in processed_observations]) 
	most_common_label = counter.most_common(1)[0][0]
	return most_common_label

def score(y_predict, y):
	# print(y_predict)
	# print(y)
	print(len(np.where(y_predict==y)[0]))
	return len(np.where(y_predict==y)[0]) / len(y)

X, y = read_data(base_path, 'diabetes.csv')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# fit(X_train.values, y_train.values)
# predictions = []
# for x_test in X_test.values:
# 	prediction = predict(x_test, 10)
# 	predictions.append(prediction)

# print(score(np.array(predictions), y_test.values))


from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
predicts = neigh.predict(X_test)

print(neigh.score(y_test, predicts))














