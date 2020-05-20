import tensorflow as tf, numpy as np, data, os, math, pickle, sys, matplotlib.pyplot as plt
from pdb import set_trace
from datetime import datetime
from tensorflow.keras import backend as K
from classes import modelSet, modelDictVal

huberDelta = .5

def smoothL1(y_true, y_pred): # custom loss function for unit vector output
	x = tf.keras.backend.abs(y_true - y_pred)
	x = tf.where(x < huberDelta, 0.5 * x ** 2, huberDelta * (x - 0.5 * huberDelta))
	return	tf.keras.backend.sum(x)

def coordsOutPut(x): # add coordinate output layer
	coords = tf.keras.layers.Conv2D(18, (1,1), name = 'coordsOut', kernel_initializer = tf.keras.initializers.GlorotUniform(seed=0), padding = 'same')(x)
	#coords = tf.keras.layers.BatchNormalization(name = 'batchCoords')(coords)
	#coords = tf.keras.layers.Activation('relu')(coords)
	return coords
	
def classOutput(x): # add class output layer
	classPred = tf.keras.layers.Conv2D(1, (1,1), name = 'classConv', kernel_initializer = tf.keras.initializers.GlorotUniform(seed=0), padding = 'same')(x)
	classPred = tf.keras.layers.BatchNormalization(name = 'classBatch')(classPred)
	classPred = tf.keras.layers.Activation('relu', name = "classOut")(classPred)
	
	return classPred

def convLayer(x, numFilters, kernelSize, strides = 1, dilation = 1):
	x = tf.keras.layers.Conv2D(numFilters, kernelSize, strides = strides, kernel_initializer = tf.keras.initializers.GlorotUniform(seed=0), padding = 'same', dilation_rate = dilation)(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.Activation('relu')(x)
	
	return x
	
def stvNet(inputShape = (480, 640, 3), outVectors = True, outClasses = True, modelName = "stvNet"):
	
	xIn = tf.keras.Input(inputShape, dtype = np.dtype('uint8'))
	
	x = tf.keras.layers.Lambda(lambda x: x / 255) (xIn)
	
	x = tf.keras.layers.Conv2D(64, 7, input_shape = inputShape, kernel_initializer = tf.keras.initializers.GlorotUniform(seed=0), padding = 'same')(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.Activation('relu')(x)
	
	res1 = x
	
	x = tf.keras.layers.MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(x)
	
	skip = x
	
	x = convLayer(x, 64, 3)
	x = convLayer(x, 64, 3)
	
	x = tf.keras.layers.Add()([x, skip])
	skip = x
	
	x = convLayer(x, 64, 3)
	x = convLayer(x, 64, 3)
	
	x = tf.keras.layers.Add()([x, skip])
	skip = tf.keras.layers.MaxPool2D(pool_size = 2, padding = 'same')(x)
	skip = tf.pad(skip, [[0, 0], [0, 0], [0, 0], [32, 32]]) # linear projection
	res2 = x
	
	x = convLayer(x, 128, 3, 2)
	x = convLayer(x, 128, 3)
	
	x = tf.keras.layers.Add()([x, skip])
	skip = x
	
	x = convLayer(x, 128, 3)
	x = convLayer(x, 128, 3)
	
	x = tf.keras.layers.Add()([x, skip])
	skip = tf.keras.layers.MaxPool2D(pool_size = 2, padding = 'same')(x)
	skip = tf.pad(skip, [[0, 0], [0, 0], [0, 0], [64, 64]]) # linear projection
	res3 = x
	
	x = convLayer(x, 256, 3, 2)
	x = convLayer(x, 256, 3)
	
	x = tf.keras.layers.Add()([x, skip])
	skip = x
	
	x = convLayer(x, 256, 3)
	x = convLayer(x, 256, 3)
	
	x = tf.keras.layers.Add()([x, skip])
	skip = tf.pad(x, [[0, 0], [0, 0], [0, 0], [128, 128]])
	res4 = x
	
	x = convLayer(x, 512, 3, dilation = 2)
	x = convLayer(x, 512, 3, dilation = 2)
	
	x = tf.keras.layers.Add()([x, skip])
	skip = x
	
	x = convLayer(x, 512, 3, dilation = 2)
	x = convLayer(x, 512, 3, dilation = 2)
	
	x = tf.keras.layers.Add()([x, skip])
	
	x = convLayer(x, 256, 3)
	x = convLayer(x, 256, 3)
	
	x = tf.keras.layers.Add()([x, res4])
	x = tf.keras.layers.UpSampling2D()(x)
	
	x = convLayer(x, 128, 3)
	
	x = tf.keras.layers.Add()([x, res3])
	x = tf.keras.layers.UpSampling2D()(x)
	
	x = convLayer(x, 64, 3)
	
	x = tf.keras.layers.Add()([x, res2])
	x = tf.keras.layers.UpSampling2D()(x)
	
	x = convLayer(x, 32, 3)
	
	outputs = []
	
	if outVectors:
		outputs.append(coordsOutPut(x))
	if outClasses:
		outputs.append(classOutput(x))
	
	return tf.keras.Model(inputs = xIn, outputs = outputs, name = modelName)

def stvNetNew(inputShape = (480, 640, 3), outVectors = True, outClasses = True, modelName = "stvNetNew"):
	
	xIn = tf.keras.Input(inputShape, dtype = np.dtype('uint8'))
	
	x = tf.keras.layers.Lambda(lambda x: x / 255) (xIn)
	
	x = tf.keras.layers.Conv2D(64, 7, input_shape = inputShape, kernel_initializer = tf.keras.initializers.GlorotUniform(seed=0), padding = 'same')(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.Activation('relu')(x)
	
	res1 = x
	
	x = tf.keras.layers.MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(x)
	
	skip = x
	
	x = convLayer(x, 64, 3)
	x = convLayer(x, 64, 3)
	
	x = tf.keras.layers.Add()([x, skip])
	skip = x
	
	x = convLayer(x, 64, 3)
	x = convLayer(x, 64, 3)
	
	x = tf.keras.layers.Add()([x, skip])
	skip = tf.keras.layers.MaxPool2D(pool_size = 2, padding = 'same')(x)
	skip = tf.pad(skip, [[0, 0], [0, 0], [0, 0], [32, 32]]) # linear projection
	res2 = x
	
	x = convLayer(x, 128, 3, 2)
	x = convLayer(x, 128, 3)
	
	x = tf.keras.layers.Add()([x, skip])
	skip = x
	
	x = convLayer(x, 128, 3)
	x = convLayer(x, 128, 3)
	
	x = tf.keras.layers.Add()([x, skip])
	skip = tf.keras.layers.MaxPool2D(pool_size = 2, padding = 'same')(x)
	skip = tf.pad(skip, [[0, 0], [0, 0], [0, 0], [64, 64]]) # linear projection
	res3 = x
	
	x = convLayer(x, 256, 3, 2)
	x = convLayer(x, 256, 3)
	
	x = tf.keras.layers.Add()([x, skip])
	skip = x
	
	x = convLayer(x, 256, 3)
	x = convLayer(x, 256, 3)
	
	x = tf.keras.layers.Add()([x, skip])
	skip = tf.pad(x, [[0, 0], [0, 0], [0, 0], [128, 128]])
	res4 = x
	
	x = convLayer(x, 512, 3, dilation = 2)
	x = convLayer(x, 512, 3, dilation = 2)
	
	x = tf.keras.layers.Add()([x, skip])
	skip = x
	
	x = convLayer(x, 512, 3, dilation = 2)
	x = convLayer(x, 512, 3, dilation = 2)
	
	x = tf.keras.layers.Add()([x, skip])
	
	x = convLayer(x, 256, 3)
	x = convLayer(x, 256, 3)
	
	x = tf.keras.layers.Add()([x, res4])
	x = tf.keras.layers.UpSampling2D()(x)
	
	x = convLayer(x, 128, 3)
	
	x = tf.keras.layers.Add()([x, res3])
	x = tf.keras.layers.UpSampling2D()(x)
	
	x = convLayer(x, 64, 3)
	
	x = tf.keras.layers.Add()([x, res2])
	x = tf.keras.layers.UpSampling2D()(x)
	
	x = tf.keras.layers.Add()([x, res1])
	
	x = convLayer(x, 32, 3)
	
	outputs = []
	
	if outVectors:
		outputs.append(coordsOutPut(x))
	if outClasses:
		outputs.append(classOutput(x))
	
	return tf.keras.Model(inputs = xIn, outputs = outputs, name = modelName)
	
def uNet(inputShape = (480, 640, 3), outVectors = True, outClasses = True, modelName = "uNet"): # neural net structure used for image segmentation
	xIn = tf.keras.Input(inputShape, dtype = np.dtype('uint8'))
	
	x = tf.keras.layers.Lambda(lambda x: x / 255) (xIn)
	
	c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (x)
	c1 = tf.keras.layers.Dropout(0.1) (c1)
	c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
	p1 = tf.keras.layers.MaxPool2D((2, 2)) (c1)

	c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
	c2 = tf.keras.layers.Dropout(0.1) (c2)
	c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
	p2 = tf.keras.layers.MaxPool2D((2, 2)) (c2)

	c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
	c3 = tf.keras.layers.Dropout(0.2) (c3)
	c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
	p3 = tf.keras.layers.MaxPool2D((2, 2)) (c3)

	c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
	c4 = tf.keras.layers.Dropout(0.2) (c4)
	c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
	p4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2)) (c4)

	c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
	c5 = tf.keras.layers.Dropout(0.3) (c5)
	c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

	u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
	u6 = tf.keras.layers.concatenate([u6, c4])
	c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
	c6 = tf.keras.layers.Dropout(0.2) (c6)
	c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

	u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
	u7 = tf.keras.layers.concatenate([u7, c3])
	c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
	c7 = tf.keras.layers.Dropout(0.2) (c7)
	c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

	u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
	u8 = tf.keras.layers.concatenate([u8, c2])
	c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
	c8 = tf.keras.layers.Dropout(0.1) (c8)
	c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

	u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
	u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
	c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
	c9 = tf.keras.layers.Dropout(0.1) (c9)
	c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

	outputs = []
	if outVectors:
		#outputs.append(tf.keras.layers.Conv2D(18, (1, 1), activation='sigmoid') (c9))
		outputs.append(tf.keras.layers.Conv2D(18, (1,1), kernel_initializer = tf.keras.initializers.GlorotUniform(seed=0), padding = 'same') (c9))
		#outputs.append(coordsOutPut(c9))
	if outClasses:
		outputs.append(tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid') (c9))
		#outputs.append(classOutput(c9))
		#outputs.append(tf.keras.layers.Conv2D(1, (1,1), name = 'classConv', kernel_initializer = tf.keras.initializers.GlorotUniform(seed=0), padding = 'same', activation = tf.keras.layers.Activation('relu')) (c9))
	
	return tf.keras.Model(inputs = [xIn], outputs = outputs, name = modelName)
	
def trainModel(modelStruct, modelGen, modelClass = 'cat', batchSize = 2, optimizer = tf.keras.optimizers.Adam, learning_rate = 0.01, losses = None, metrics = ['accuracy'], saveModel = True, modelName = 'stvNet_weights', epochs = 1, loss_weights = None, outVectors = False, outClasses = False, dataSplit = True, altLabels = True, augmentation = True): # train and save model weights
	if not (outVectors or outClasses):
		print("At least one of outVectors or outClasses must be set to True.")
		return
	model = modelStruct(outVectors = outVectors, outClasses = outClasses, modelName = modelName)
	#model.summary()
	model.compile(optimizer = optimizer(learning_rate = learning_rate), loss = losses, metrics = metrics, loss_weights = loss_weights)
	
	trainData, validData = None, None
	if dataSplit: # if using datasplit, otherwise all available data is used
		trainData, validData = data.getDataSplit(modelClass = modelClass)
	
	logger = tf.keras.callbacks.CSVLogger("models\\history\\" + modelName + "_" + modelClass + "_history.csv", append = True)
	#evalLogger = tf.keras.callbacks.CSVLogger("models\\history\\" + modelName + "_" + modelClass + "_eval_history.csv", append = True)
	
	history, valHistory = [], []
	
	if type(losses) is dict:
		outKeys = list(losses.keys())
		if len(outKeys) == 2: # combined output
			for i in range(epochs):
				print("Epoch {0} of {1}".format(i + 1, epochs))
				hist = model.fit(modelGen(modelClass, batchSize, masterList = trainData, out0 = outKeys[0], out1 = outKeys[1], altLabels = altLabels, augmentation = augmentation), steps_per_epoch = math.ceil(len(trainData) / batchSize), max_queue_size = 2, callbacks = [logger])
				history.append(hist.history)
				if dataSplit:
					print("Validation:")
					valHist = model.evaluate(modelGen(modelClass, batchSize, masterList = validData, out0 = outKeys[0], out1 = outKeys[1], altLabels = altLabels, augmentation = False), steps = math.ceil(len(validData) / batchSize), max_queue_size = 2)
					valHistory.append(valHist)
		else:
			raise Exception("Probably shouldn't be here ever..")
	else:
		for i in range(epochs):
			print("Epoch {0} of {1}".format(i + 1, epochs))
			hist = model.fit(modelGen(modelClass, batchSize, masterList = trainData, altLabels = altLabels, augmentation = augmentation), steps_per_epoch = math.ceil(len(trainData) / batchSize), max_queue_size = 2, callbacks = [logger])
			history.append(hist.history)
			if dataSplit:
				print("Validation:")
				valHist = model.evaluate(modelGen(modelClass, batchSize, masterList = validData, altLabels = altLabels, augmentation = False), steps = math.ceil(len(validData) / batchSize), max_queue_size = 2)
				valHistory.append(valHist)
		
	historyLog = {"struct": modelStruct.__name__,
		"class" : modelClass,
		"optimizer": optimizer,
		"lr" : learning_rate,
		"losses": losses,
		"name": modelName,
		"epochs": epochs,
		"history": history,
		"evalHistory": valHistory,
		"timestamp": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
	}
	
	if saveModel:
		model.save_weights(os.path.dirname(os.path.realpath(__file__)) + '\\models\\' + modelName + '_' + modelClass)
		model.save(os.path.dirname(os.path.realpath(__file__)) + '\\models\\' + modelName + '_' + modelClass)
		if not os.path.exists("models\\history\\" + modelName + '_trainHistory'):
			with open("models\\history\\" + modelName + '_' + modelClass + '_trainHistory', 'wb') as f: # create model history
				pickle.dump([], f)
		with open("models\\history\\" + modelName + '_' + modelClass + '_trainHistory', 'rb') as f: # loading old history 
			histories = pickle.load(f)
		histories.append(historyLog)
		with open("models\\history\\" + modelName + '_' + modelClass + '_trainHistory', 'wb') as f: # saving the history of the model
			pickle.dump(histories, f)
		
	return model

def trainModels(modelSets, shutDown = False):
	for modelSet in modelSets:
		print("Training {0}".format(modelSet.name))
		model = modelsDict[modelSet.name]
		trainModel(model.structure, model.generator, modelClass = modelSet.modelClass, epochs = model.epochs, losses = model.losses, modelName = modelSet.name, outClasses = model.outClasses, outVectors = model.outVectors, learning_rate = model.lr, metrics = model.metrics, altLabels = model.altLabels, augmentation = model.augmentation)
		
		K.clear_session()
		K.reset_uids()
	
	if shutDown:
		os.system('shutdown -s')

def evaluateModel(modelStruct, modelName, evalGen, modelClass = 'cat', outVectors = False, outClasses = False, batchSize = 2, optimizer = tf.keras.optimizers.Adam, learning_rate = 0.01, losses = None, metrics = ['accuracy'], samples = 100): # test existing model performance
	model = tf.keras.models.load_model(os.path.dirname(os.path.realpath(__file__)) + '\\models\\' + modelName + '_' + modelClass)
	model.evaluate(evalGen(modelClass, batchSize), steps = samples // batchSize)
	
def evaluateModels(modelSets, batchSize = 2, dataSplit = True):
	
	for modelSet in modelSets:
		validData = (data.getDataSplit(modelClass = modelSet.modelClass)[1] if dataSplit else None)
		modelEnt = modelsDict[modelSet.name]
		model = loadModelWeights(modelEnt.structure, modelSet.name, modelSet.modelClass, modelEnt.outVectors, modelEnt.outClasses, losses = modelEnt.losses, metrics = modelEnt.metrics)
		if type(model.losses) is dict:
			outKeys = list(losses.keys())
			if len(outKeys) == 2: # combined output
				model.evaluate(modelEnt.generator(modelSet.modelClass, batchSize = batchSize, masterList = validData, out0 = outKeys[0], out1 = outKeys[1], altLabels = modelEnt.altLabels, augmentation = False), steps = math.ceil(len(validData) / batchSize), max_queue_size = 2)
			else:
				raise Exception("Probably shouldn't be here ever..")
		else:
			model.evaluate(modelEnt.generator(modelSet.modelClass, batchSize = batchSize, masterList = validData, altLabels = modelEnt.altLabels, augmentation = False), steps = math.ceil(len(validData) / batchSize), max_queue_size = 2)
	
def trainModelClassGen(modelStruct, modelName, losses, modelClass = 'cat', batchSize = 2, optimizer = tf.keras.optimizers.Adam, learningRate = 0.001, metrics = ['accuracy'], epochs = 1, outVectors = False, outClasses = False, outVecName = None, outClassName = None): # simulates generator behaviour, unused
	model = modelStruct(outVectors = outVectors, outClasses = outClasses, modelName = modelName)
	#model.summary()
	model.compile(optimizer = optimizer(learning_rate = learningRate), loss = losses, metrics = metrics)
	
	myGen = generatorClass(modelClass, outVectors = outVectors, outClasses = outClasses, outVecName = outVecName, outClassName = outClassName)
	logger = tf.keras.callbacks.CSVLogger("models\\history\\" + modelName + "_" + modelClass + "_history.csv", append = True)
	
	for i in range(epochs):
		print("Epoch {0} of {1}".format(i + 1, epochs))
		while True:
			epochEnd, x, y = myGen.serveBatch()
			if epochEnd:
				break
			model.fit(x, y, callbacks = [logger], verbose = 0)
			#update_progress(myGen.i / myGen.dataLength)
			
	return model

def loadModelWeights(modelStruct, modelName, modelClass = 'cat', outVectors = False, outClasses = False, optimizer = tf.keras.optimizers.Adam, learning_rate = 0.01, losses = None, metrics = ['accuracy']): # return compiled tf keras model
	if not (outVectors or outClasses):
		raise Exception("At least one of outVectors or outClasses must be set to True.")
	model = modelStruct(outVectors = outVectors, outClasses = outClasses, modelName = modelName)
	model.load_weights(os.path.dirname(os.path.realpath(__file__)) + '\\models\\' + modelName + '_' + modelClass)
	model.compile(optimizer = optimizer(learning_rate = learning_rate), loss = losses, metrics = metrics)
	return model

def loadHistory(modelName, modelClass = 'cat'):
	with open("models\\history\\" + modelName + '_' + modelClass + '_trainHistory', 'rb') as f: # loading old history 
		histories = pickle.load(f)
		for hist in histories:
			print("Structure: {0}\nClass: {1}\nOptimizer: {2}\nLearningRate: {3}\nLosses: {4}\nName: {5}\nEpochs: {6}\nTimestamp: {7}\nTraining History:\n".format(hist['struct'], hist['class'], hist['optimizer'], hist['lr'], hist['losses'], hist['name'], hist['epochs'], hist['timestamp']))
			for i, epoch in enumerate(hist['history']):
				print("{0}: {1}".format(i, epoch))
			print("\nEvaluation History:\n")
			for i, epoch in enumerate(hist['evalHistory']):
				print("{0}: {1}".format(i, epoch))
			print("\n")

def loadHistories(modelSets):
	for modelSet in modelSets:
		print("Loading {0}".format(modelSet.name))
		loadHistory(modelSet.name, modelSet.modelClass)
		
def plotHistories(modelSets): # display loss values over epochs using pyplot
	plt.figure()
	maxLen = 0
	for modelSet in modelSets:
		with open("models\\history\\" + modelSet.name + '_' + modelSet.modelClass + '_trainHistory', 'rb') as f: # loading old history 
			histories = pickle.load(f)
		for hist in histories:
			if len(hist['history']) > maxLen:
				maxLen = len(hist['history'])
			plt.subplot(211)
			plt.plot([x['loss'] for x in hist['history']], label = hist['name'])
			plt.subplot(212)
			plt.plot([x[0] for x in hist['evalHistory']], label = hist['name'])
	plt.subplot(211)
	plt.ylabel("Training Loss")
	plt.xlabel("Epoch")
	plt.xticks(np.arange(0, maxLen, 1.0))
	plt.subplot(212)
	plt.ylabel("Validation Loss")
	plt.xlabel("Epoch")
	plt.xticks(np.arange(0, maxLen, 1.0))
	plt.legend()
	plt.show()
	plt.close()

class generatorClass: # simulates generator behaviour, unused
	
	def __init__(self, modelClass, height = 480, width = 640, batchSize = 2, outVectors = False, outClasses = False, outVecName = None, outClassName = None):
		if not (outClasses or outVectors):
			raise Exception("Must have at least one output")
		self.i = 0
		self.basePath = os.path.dirname(os.path.realpath(__file__)) + '\\LINEMOD\\' + modelClass
		self.masterList = getMasterList(self.basePath)
		self.dataLength = len(self.masterList)
		self.height = height
		self.width = width
		self.outVectors = outVectors
		self.outClasses = outClasses
		self.outVecName = outVecName
		self.outClassName = outClassName
		self.batchSize = batchSize
		
	def serveBatch(self):
		xBatch = []
		yCoordBatch = []
		yClassBatch = []
		output = {}
		
		for b in range(self.batchSize):
			if self.i == self.dataLength:
				self.i = 0
				random.shuffle(self.masterList)
				return True, [], [], []
			x = filePathToArray(self.basePath + '\\JPEGImages\\' + self.masterList[self.i][0], self.height, self.width)
			
			with open(self.basePath + '\\labels\\' + self.masterList[self.i][2]) as f:
				labels = f.readline().split(' ')[1:19]
			
			yCoordsLabels = np.zeros((self.height, self.width, 18)) # 9 coordinates
			yClassLabels = np.zeros((self.height, self.width, 1)) # 1 class confidence value per model
			
			modelMask = filePathToArray(self.basePath + '\\mask\\' + self.masterList[self.i][1], self.height, self.width)
			modelCoords = np.where(modelMask == 255)[:2]
			
			for modelCoord in zip(modelCoords[0][::3], modelCoords[1][::3]):
				setTrainingPixel(yCoordsLabels, modelCoord[0], modelCoord[1], labels, self.height, self.width)
				yClassLabels[modelCoord[0]][modelCoord[1]][0] = 1
			xBatch.append(x)
			yCoordBatch.append(yCoordsLabels)
			yClassBatch.append(yClassLabels)
			self.i += 1
			
		if self.outVectors:
			output[self.outVecName] = np.array(yCoordBatch)
		if self.outClasses:
			output[self.outClassName] = np.array(yClassBatch)
		return (False, np.array(xBatch), output)
			
modelsDict = {
	'uNet_classes' : modelDictVal(uNet, data.classTrainingGenerator, tf.keras.losses.BinaryCrossentropy(), False, True, epochs = 20, lr = 0.001, augmentation = False),
	'uNet_coords' : modelDictVal(uNet, data.coordsTrainingGenerator, tf.keras.losses.Huber(), True, False, epochs = 5, lr = 0.001, metrics = ['mae', 'mse']),
	'uNet_coords_smooth' : modelDictVal(uNet, data.coordsTrainingGenerator, smoothL1, True, False, epochs = 3, lr = 0.0001, metrics = ['mae', 'mse']),
	'stvNet' : modelDictVal(stvNet, data.combinedTrainingGenerator, {'coordsOut': tf.keras.losses.Huber(), 'classOut': tf.keras.losses.BinaryCrossentropy()}, True, True, epochs = 5, lr = 0.00005, metrics = {'coordsOut': ['mae', 'mse'], "classOut": ['accuracy']}),
	'stvNet_coords_slow_learner' : modelDictVal(stvNet, data.coordsTrainingGenerator, tf.keras.losses.Huber(), True, False, epochs = 40, lr = 0.00001, metrics = ['mae', 'mse'], outVecName = 'coordsOut'),
	'stvNetAltLabels' : modelDictVal(stvNet, data.combinedTrainingGenerator, {'coordsOut': tf.keras.losses.Huber(), 'classOut': tf.keras.losses.BinaryCrossentropy()}, True, True, epochs = 10, lr = 0.001, metrics = {'coordsOut': ['mae', 'mse'], "classOut": ['accuracy']}, altLabels = True, augmentation = True),
	'stvNetNormLabels' : modelDictVal(stvNet, data.combinedTrainingGenerator, {'coordsOut': tf.keras.losses.Huber(), 'classOut': tf.keras.losses.BinaryCrossentropy()}, True, True, epochs = 10, lr = 0.001, metrics = {'coordsOut': ['mae', 'mse'], "classOut": ['accuracy']}, altLabels = False, augmentation = True),
	'stvNet_coords' : modelDictVal(stvNet, data.coordsTrainingGenerator, tf.keras.losses.Huber(), True, False, epochs = 20, lr = 0.001, metrics = ['mae', 'mse'], altLabels = False, augmentation = True),
	'stvNet_coords_altLabels' : modelDictVal(stvNet, data.coordsTrainingGenerator, tf.keras.losses.Huber(), True, False, epochs = 20, lr = 0.001, metrics = ['mae', 'mse'], altLabels = True, augmentation = True),
	'stvNet_coords_altLabels_noAug' : modelDictVal(stvNet, data.coordsTrainingGenerator, tf.keras.losses.Huber(), True, False, epochs = 20, lr = 0.001, metrics = ['mae', 'mse'], altLabels = True, augmentation = False),
	'stvNet_coords_noAug' : modelDictVal(stvNet, data.coordsTrainingGenerator, tf.keras.losses.Huber(), True, False, epochs = 20, lr = 0.001, metrics = ['mae', 'mse'], altLabels = False, augmentation = False),
	'stvNet_classes' : modelDictVal(stvNet, data.classTrainingGenerator, tf.keras.losses.BinaryCrossentropy(), False, True, epochs = 10, lr = 0.001, altLabels = False, augmentation = True),
	'stvNet_classes_noAug' : modelDictVal(stvNet, data.classTrainingGenerator, tf.keras.losses.BinaryCrossentropy(), False, True, epochs = 10, lr = 0.001, altLabels = False, augmentation = False),
	'stvNet_new_coords_alt' : modelDictVal(stvNetNew, data.coordsTrainingGenerator, tf.keras.losses.Huber(), True, False, epochs = 20, lr = 0.001, metrics = ['mae', 'mse'], altLabels = True, augmentation = False),
	'stvNet_new_coords' : modelDictVal(stvNetNew, data.coordsTrainingGenerator, tf.keras.losses.Huber(), True, False, epochs = 20, lr = 0.001, metrics = ['mae', 'mse'], altLabels = False, augmentation = False),
	'stvNet_new_coords_alt_aug' : modelDictVal(stvNetNew, data.coordsTrainingGenerator, tf.keras.losses.Huber(), True, False, epochs = 20, lr = 0.001, metrics = ['mae', 'mse'], altLabels = True, augmentation = True),
	'stvNet_new_coords_aug' : modelDictVal(stvNetNew, data.coordsTrainingGenerator, tf.keras.losses.Huber(), True, False, epochs = 20, lr = 0.001, metrics = ['mae', 'mse'], altLabels = False, augmentation = True),
	'stvNet_new_classes' : modelDictVal(stvNetNew, data.classTrainingGenerator, tf.keras.losses.BinaryCrossentropy(), False, True, epochs = 20, lr = 0.001, augmentation = False),
	'stvNet_new_combined' : modelDictVal(stvNetNew, data.combinedTrainingGenerator, {'coordsOut': tf.keras.losses.Huber(), 'classOut': tf.keras.losses.BinaryCrossentropy()}, True, True, epochs = 20, lr = 0.001, metrics = {'coordsOut': ['mae', 'mse'], "classOut": ['accuracy']}, augmentation = False),
}
	
if __name__ == "__main__" :
	#modelSets = [modelSet('stvNet_coords_altLabels'), modelSet('stvNet_coords_noAug'), modelSet('stvNet_coords_altLabels_noAug'), modelSet('stvNet_coords'), modelSet('stvNet_new_coords_alt'), modelSet('stvNet_new_coords_aug'), modelSet('stvNet_new_coords')] # vector outputs
	#modelSets = [modelSet('stvNet_new_classes'), modelSet('uNet_classes'), modelSet('stvNet_classes')] # class outputs
	modelSets = [modelSet('stvNet_new_combined')]
	
	#modelSets = [modelSet('stvNet_new_coords_alt')]
	
	#evaluateModels(modelSets)
	#loadHistories(modelSets)
	trainModels(modelSets)
	#plotHistories(modelSets)
	pass