'''
@author: Yuliang Li
@Contact: yuliang.li@gatech.edu
@Summary: ML4T Project 3: machine learner for stock price forecasting
'''

import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da

import datetime as dt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from RandomForestLearner import *
from LinRegLearner import *

def RMSE(DTest, DLearn):
	'''
	Root-Mean-Square Error
	@DTest: tested correct data
	@Dlearn: data form learning
	'''
	row_queryr = float(DLearn.shape[0])
	RMSE = np.sum(np.power((DTest - DLearn), 2.0)) / row_queryr
	RMSE = np.sqrt(RMSE)
	return RMSE
#end RMSE

def TrainGenerator():
	''' 
	Input raw training data files -> get the actual close price -> generate feature and Y value 
	'''

	# Input the raw files and get the close price
	c_dataobj = da.DataAccess('Yahoo')
	
	dt_start = dt.datetime(2000, 2, 1)
	dt_end = dt.datetime(2012, 9, 13)
	dt_timeofday = dt.timedelta(hours = 16)
	ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt_timeofday)

	train_close = list()
	
	for i in range(0, 100):
		if i < 10:
			filename = '00' + str(i)
		else:
			filename = '0' + str(i)
		ls_symbol = ['ML4T-' + filename]
		#print ls_symbol
		
		ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
		
		#print "Lord of Ring"

		ldf_data = c_dataobj.get_data(ldt_timestamps, ls_symbol, ls_keys)
	
		data_train = dict(zip(ls_keys, ldf_data))
		df_close = data_train['actual_close']
		train_close.append(df_close['ML4T-'+filename])

	print len(train_close)
	# Feature
	OvernightDiv = np.zeros([700 * len(train_close), 1]) # one day price change
	OnChangeDiv = np.zeros([700 * len(train_close), 1]) # difference between today and yesterday price change 
	# Y value
	FivChangeDiv = np.zeros([700 * len(train_close), 1]) # price after 5 days
	count = 0

	# Generate the feature 
	for PClose in train_close:
		for dat in range(100, 800):
			FivChangeDiv[dat - 100 + 700 * count] = PClose[dat+5] - PClose[dat]
			OvernightDiv[dat - 100 + 700 * count] = PClose[dat+1] - PClose[dat]
			OnChangeDiv[dat - 100 + 700 * count] = PClose[dat+2] - 2*PClose[dat+1] + PClose[dat]
		count = count + 1
	#end for

	FeatureData = np.zeros([700 * len(train_close), 2])
	FeatureData[:, 0] = OvernightDiv[:, 0]
	FeatureData[:, 1] = OnChangeDiv[:, 0]
	
	return FeatureData, FivChangeDiv
#end TrainGenerator

def TestGenerator(df_close):
	''' 
	Input raw test data files -> get the actual close price -> generate feature and Y value
	@df_close: actual close price of test data set
	'''

	# Feature
	OvernightDiv = np.zeros([len(df_close)-5, 1]) # one day price change
	OnChangeDiv = np.zeros([len(df_close)-5, 1]) # difference between today and yesterday price change 
	# Y value
	FivChangeDiv = np.zeros([len(df_close)-5, 1]) # price after 5 days

	# Generate the feature 
	for dat in range(0, len(df_close)-5):
		FivChangeDiv[dat] = df_close[dat+5] - df_close[dat]
		OvernightDiv[dat] = df_close[dat+1] - df_close[dat]
		OnChangeDiv[dat] = df_close[dat+2] - 2*df_close[dat+1] + df_close[dat]
	#end for

	FeatureData = np.zeros([len(df_close)-5, 2])
	FeatureData[:, 0] = OvernightDiv[:, 0]
	FeatureData[:, 1] = OnChangeDiv[:, 0]
	
	return FeatureData, FivChangeDiv
#end TestGenerator

def LinRegTest(XTrain, YTrain, close, filename):
	'''
	Using RandomForest learner to predict how much the price will change in 5 days
	@filename: the file's true name is ML4T-filename
	@XTrain: the train data for feature
	@YTrain: the train data for actual price after 5 days
	@close: the actual close price of Test data set
	@k: the number of trees in the forest
	'''
	
	XTest, YTest = TestGenerator(close)

	#plot thge feature
	plt.clf()
	fig = plt.figure()
	fig.suptitle('The value of features')
	plt.plot(range(100), XTest[0:100, 0], 'b', label = 'One day price change')
	plt.plot(range(100), XTest[0:100, 1], 'r', label = 'difference between two day price change')
	plt.legend(loc = 4)
	plt.ylabel('Price')
	filename4 = 'feature' + filename + '.pdf'
	fig.savefig(filename4, format = 'pdf')

	LRL = LinRegLearner()
	cof = LRL.addEvidence(XTrain, YTrain)
	YLearn = LRL.query(XTest, cof)
	return YLearn
#end RandomForestTest

def PricePredict(filename):
	'''
	Predict the stock's price after 5 days
	@filename: the file's true name is ML4T-filename
	'''

	# Training data
	FeatureData, PricePred = TrainGenerator()

	# get the test data
	# Input the raw files and get the close price
	c_dataobj = da.DataAccess('Yahoo')	
	dt_start = dt.datetime(2000, 2, 1)
	dt_end = dt.datetime(2012, 9, 13)
	dt_timeofday = dt.timedelta(hours = 16)
	ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt_timeofday)
	ls_symbol = ['ML4T-' + filename]
	ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
	ldf_data = c_dataobj.get_data(ldt_timestamps, ls_symbol, ls_keys)
	data_train = dict(zip(ls_keys, ldf_data))
	df_close = data_train['actual_close']
	df_close = df_close['ML4T-' + filename]
	#print len(df_close)

	PriceChange = LinRegTest(FeatureData, PricePred, df_close, filename)
	#print len(PriceChange)

	#print PriceChange
	df_close = np.array(df_close)
	#print df_close
	#df_close_learn = np.zeros([len(df_close)-5, 1])
	#df_close_last = np.zeros([len(df_close)-5, 1])
	df_close_last = df_close[5:len(df_close)]
	df_close_learn = df_close[0:len(df_close)-5]
	Price_true = np.zeros([len(df_close_last), 1]) # actual price 
	Price_Pre = np.zeros([len(df_close_learn), 1]) # predicted price
	#print Price_Pre.size.shape
	Price_true[:, 0] = df_close_last.T
	Price_Pre[:, 0] = df_close_learn.T
	#print Price_Pre
	Price_Pre = Price_Pre + PriceChange

	return Price_true, Price_Pre
#end PricePredict

def Test_Plot(YTest, YLearn, filename):
	'''
	RMS Error measure and plot
	@YTest: actual test data
	@YLearn: data from learning
	'''
	print "RMS Error is ", RMSE(YTest, YLearn)

	plt.clf()  
	fig = plt.figure()
	fig.suptitle('The first 100 days of Yactual and Ypredict')
	plt.plot(range(100), YTest[0:100, :], 'b', label = 'Yactual')
	plt.plot(range(100), YLearn[0:100, :], 'r', label = 'Ypredict')
	plt.legend()
	plt.ylabel('price')
	filename1 = 'first-100-compare'+filename+'.pdf'
	fig.savefig(filename1, format = 'pdf')

	plt.clf()
	fig = plt.figure()
	fig.suptitle('The last 100 days of Yactual and Ypredict')
	plt.plot(range(100), YTest[-100:, :], 'b', label = 'Yactual')
	plt.plot(range(100), YLearn[-100:, :], 'r', label = 'Ypredict')
	plt.legend()
	plt.ylabel('price')
	filename2 = 'last-100-compare' + filename+ '.pdf'
	fig.savefig(filename2, format = 'pdf')

	plt.clf()
	fig = plt.figure()
	fig.suptitle('Scatterplot of Ypredict versus Yactual')
	plt.scatter(YTest, YLearn)
	plt.xlabel('YTest Price')
	plt.ylabel('YLearn Price')
	filename3 = 'test-learn-compare' + filename +'.pdf'
	fig.savefig(filename3, format = 'pdf')



if __name__ == '__main__':
	ActualPrice, ForePrice = PricePredict('292')
	Test_Plot(ActualPrice, ForePrice, '292')

	ActualPrice, ForePrice = PricePredict('151') # Yuliang Li
	Test_Plot(ActualPrice, ForePrice, '151')



