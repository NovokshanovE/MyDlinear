from pandas import read_csv
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from matplotlib import pyplot
from numpy import mean
# load data
series = read_csv('ETTh1.csv', header=0, index_col=0,usecols=['date', 'HUFL'])
# prepare data

print(series)
X = series.values
train, test = X[0:-24], X[-24:]
window_sizes = range(1, 25)
scores = list()
for w in window_sizes:
	# walk-forward validation
	history = [x for x in train]
	predictions = list()
	for i in range(len(test)):
		# make prediction
		yhat = mean(history[-w:])
		predictions.append(yhat)
		# observation
		history.append(test[i])
	# report performance
	rmse = sqrt(mean_absolute_error(test, predictions))
	scores.append(rmse)
	print('w=%d RMSE:%.3f' % (w, rmse))
# plot scores over window sizes values
pyplot.plot(window_sizes, scores)
pyplot.show()