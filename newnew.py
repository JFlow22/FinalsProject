
# LSTM for international airline passengers problem with memory
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)


def create_Repdataset(dataset, g, look_back):
	dataX=[]
	print('data len')
	print(len(dataset))
	leng = len(dataset) - look_back - 1
	if(g == 0):
		leng = len(dataset)


	for i in range(leng):
		a = dataset[len(dataset)-look_back:(len(dataset)), 0]
		dataX.append(a)

	return numpy.array(dataX)
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset
dataframe = read_csv('Positive-tweets-ratio.csv', usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# split into train and test sets
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
print(train)
# reshape into X=t and Y=t+1
look_back = 3
trainX, trainY = create_dataset(train, look_back)


print('the trainU is:')
#print(trainU)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]

trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))

print('the trainX is:')
print(trainX)


# create and fit the LSTM network
batch_size = 1
model = Sequential()
model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

for i in range(100):
	model.fit(trainX, trainY, epochs=5, batch_size=batch_size, verbose=2, shuffle=False)
	model.reset_states()
# make predictions

print('lol betweren')
trainPredict = model.predict(trainX, batch_size=batch_size)
model.reset_states()

print('lol before')

print('@@@@@@@@@@@@@@@@train is:  @@@@@@@@@@@@@@@@')


trainZ=(trainPredict[-6:-1])
print(trainZ)
print('@@@@@@@@@@@@@@@@trainZ is:  -----------------------')
trainZ=create_Repdataset(trainZ, 1,look_back)
print(trainZ)

print('stam lol')

trainZ = numpy.reshape(trainZ, (trainZ.shape[0], trainZ.shape[1], 1))
print(trainZ)
#trainPredict2=model.predict(trainZ, batch_size=batch_size)



future=[]
temp=(trainPredict[-5:-1])
Ather_temp=(trainPredict[-6:-1])
#temp = create_Repdataset(temp, 1, look_back)
#temp = numpy.reshape(temp, (temp.shape[0], temp.shape[1], 1))

print('temp is')
print(temp)
for i in range(35):
	print('inside the loop man!!--------------     '+str(i))
	trainZ=model.predict(trainZ, batch_size=batch_size)
	model.reset_states()
	future.append(trainZ[0][0])
	print(trainZ[0][0])
	print('lul')
	print(temp)
	temp = (Ather_temp[1:])
	temp = (numpy.append(temp, trainZ[0][0])).reshape(-1, 1)
	Ather_temp=temp
	print('Auther temp')
	print(Ather_temp)
	temp = create_Repdataset(temp, 1, look_back)
	temp = numpy.reshape(temp, (temp.shape[0], temp.shape[1], 1))
	trainZ=temp
	#model.reset_states()

	print('@@@@@@@@@@@@@@@@trainZ is:  -----------------------')
	#trainZ = (trainZ[-12:-1])

	#trainZ = create_Repdataset(trainZ,0, look_back)
	print(temp)
	#trainZ = numpy.reshape(trainZ, (trainZ.shape[0], trainZ.shape[1], 1))

print('the future is:')
print((numpy.asarray(future)).reshape(-1, 1))

testPredict = model.predict(testX, batch_size=batch_size)
print('the test predict is')
print(testPredict)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform((numpy.asarray(future)).reshape(-1, 1))
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()