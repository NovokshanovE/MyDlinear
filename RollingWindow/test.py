from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from numpy import mean
from matplotlib import dates
# load data
series = read_csv('ETTh1.csv', usecols=['date', 'HUFL'])
# prepare data
X = series.values[:,1]
data = series.values[:,0,]

w = 4
fmt = dates.DateFormatter('%H:%M:%S')
train, test = X[0:2400:24], X[2400::24]
data_test = data[2400::24,]
print(data_test)
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
pyplot.plot(test, label='реальные значения')
pyplot.plot(predictions, label='прогноз')
pyplot.xlabel("Дни")
pyplot.ylabel("HUFL")
pyplot.savefig(f"w{w}", dpi=200)
pyplot.show()