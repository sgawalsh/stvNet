class modelSet:
	
	def __init__(self, modelName, modelClass = 'cat'):
		self.name = modelName
		self.modelClass = modelClass

class modelWrapper: # allows user to use single-net and double-net model interchangeably 
	
	def __init__(self, models, bb8Labels):
		if type(models) is dict:
			if len(models.keys()) == 2:
				self.classModel = models['classModel']
				self.vecModel = models['vecModel']
				self.combined = False
			else:
				raise Exception("Probably shouldn't be here ever..")
		else:
			self.combModel = models
			self.combined = True
		
		self.bb8Labels = bb8Labels
	
	def genPredict(self, input):
		if self.combined:
			pred = self.combModel.predict(input)
			return pred[0], pred[1]
		else:
			return self.vecModel.predict(input), self.classModel.predict(input)

class modelDictVal:
	
	def __init__(self, structure, generator, losses, outVectors, outClasses, epochs = 3, lr = 0.01, metrics = ['accuracy'], outVecName = None, outClassName = None, bb8Labels = True, augmentation = True):
		self.structure = structure
		self.generator = generator
		self.losses = losses
		self.outVectors = outVectors
		self.outClasses = outClasses
		self.epochs = epochs
		self.metrics = metrics
		self.lr = lr
		self.outVecName = outVecName
		self.outClassName = outClassName
		self.bb8Labels = bb8Labels
		self.augmentation = augmentation