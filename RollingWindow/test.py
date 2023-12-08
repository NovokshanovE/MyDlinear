from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from numpy import mean
# load data
series = read_csv('ETTh1.csv', header=0, index_col=0,usecols=['date', 'HUFL'])
# prepare data
X = series.values
w = 40
train, test = X[0:-2400:10], X[-2400::10]
# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
	# make prediction
	yhat = mean(history[-w:])
	predictions.append(yhat)
	# observation
	history.append(test[i])
# plot predictions vs observations
pyplot.plot(test)
pyplot.plot(predictions)
pyplot.savefig(f"w{w}", dpi=200)
pyplot.show()