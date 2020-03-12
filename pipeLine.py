import models, os, random, data, numpy as np, matplotlib.pyplot as plt, tensorflow as tf, math, cv2, json, pickle, statistics
from pdb import set_trace
from PIL import Image
from skimage.io import imshow, imread
from classes import modelSet, modelWrapper

numHypotheses = 50 #  hypotheses considered for each keypoint
ransacThreshold = .99 # min value for population sample to agree with proposed hypothesis
maskThreshold = .9 # min value for pixel to be included in population of ransac process
pruneBool = True # flag to perform pruning operation
pruneRatio = .5 # percent of smallest weighted hypotheses to be pruned
noiseScale = .1 # artificially add noise to true target data, used for testing pnp accuracy
minHyps = 55
checkQuads = True

def predictPose(coords, classes, showClassPred = False, labels = False, addNoise = False, modelName = 'cat', checkPreds = False, altLabels = True, pruning = True):
	if showClassPred: # display class prediction
		showImage(classes)

	population = np.where(classes > maskThreshold)[:2]
	if not len(population):
		return False, None
	
	population = list(zip(population[0], population[1])) # y, x format
	
	if labels is not False: # fudge data to test pnp function
		for modelCoord in population:
			data.setTrainingPixel(coords, modelCoord[0], modelCoord[1], labels, coords.shape[0], coords.shape[1])
			if addNoise:
				coords[modelCoord[0]][modelCoord[1]] += np.random.normal(0, noiseScale, 18)
	
	hypDict = ransacVoting(population, coords)
	
	if pruning:
		pruneHypsRatio(hypDict)
		#pruneHypsStdDev(hypDict)
		pass
	
	meanDict = getMean(hypDict)
	
	#covarDict = getCovariance(hypDict, meanDict)
	if altLabels:
		pts3d = np.loadtxt(os.path.join(os.path.dirname(os.path.realpath(__file__)) + '\\LINEMOD\\' + modelName + '\\', 'altPoints.txt'))
		preds = dictToArray(meanDict)
	else:
		pts3d = np.loadtxt(os.path.join(os.path.dirname(os.path.realpath(__file__)) + '\\LINEMOD\\' + modelName + '\\', 'bb8_3d.txt'))
		preds = dictToArray(meanDict)[1:] # ignoring centroid prediction
	
	if checkPreds is not False: # show predicted keypoints on image
		labelList = data.labelFloatsToPixels(labels)
		for ind in range(len(preds)):
			px = labelList[ind + 1][0]
			py = labelList[ind + 1][1]
			print("keypoint at " + str((px, py)))
			temp = np.array(checkPreds[py][px])
			checkPreds[py][px] = np.array([0,0,0])
			#plt.figure()
			#imshow(np.squeeze(checkPreds))
			#plt.show()
			checkPreds[py][px] = temp
			
	drawPoints = np.loadtxt(os.path.join(os.path.dirname(os.path.realpath(__file__)) + '\\LINEMOD\\' + modelName + '\\', 'bb8_3d.txt'))
	
	return True, pnp(pts3d, preds, drawPoints)
	
def ransacVoting(population, coords): # ransac voting to generate 2d keypoint hypotheses
	hypDict = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
	for n in range(numHypotheses): #take two pixels, find intersection of unit vectors
		#print(n)
		p1 = population.pop(random.randrange(len(population)))
		v1 = coords[p1[0]][p1[1]]
		p2 = population.pop(random.randrange(len(population)))
		v2 = coords[p2[0]][p2[1]]
		#print(p1, p2)
		#print(v1, v2)
		for i in range(9): # find lines intersection, use as hypothesis
			m1 = v1[i * 2 + 1] / v1[i * 2] # get slopes
			m2 = v2[i * 2 + 1] / v2[i * 2]
			if not (m1 - m2): # lines must intersect
				print('slope cancel')
				continue
			b1 = p1[0] - p1[1] * m1 # get y intercepts
			b2 = p2[0] - p2[1] * m2
			x = (b2 - b1) / (m1 - m2)
			y = m1 * x + b1
			if checkQuads and (y >= p1[0] != v1[i * 2 + 1] < 0 or x >= p1[1] != v1[i * 2] < 0 or y >= p2[0] != v2[i * 2 + 1] < 0 or x >= p2[1] != v2[i * 2] < 0): # check if line intersection takes place according to unit vector directions
				continue
			#print(y, x)
			weight = 0
			for voter in population: # voting for fit of hypothesis
				yDiff = y - voter[0]
				xDiff = x - voter[1]
				
				mag = math.sqrt(yDiff ** 2 + xDiff ** 2)
				vec = coords[voter[0]][voter[1]][i * 2: i * 2 + 2]
				
				if ransacVal(yDiff / mag, xDiff / mag, vec) > ransacThreshold:
					weight += 1
			hypDict[i].append(((y, x), weight))
			
		population.append(p1)
		population.append(p2)

		
	return hypDict
	
def ransacVal(y1, x1, v2): # dot product of unit vectors to find cos(theta difference)
	v2 = v2 / np.linalg.norm(v2)
	
	return y1 * v2[1] + x1 * v2[0]
	
def pruneHypsRatio(hypDict): # prune generated hypotheses by eliminating lowest n %
	for key, hyps in hypDict.items():
		hyps.sort(key = lambda x : x[1], reverse = True)
		hypDict[key] = hyps[:round(len(hyps) * pruneRatio)]
		
def pruneHypsStdDev(hypDict, m = 2): # prune generated hypotheses using mean and stdDev
	for key, hyps in hypDict.items():
		yVals, xVals = [x[0][0]for x in hyps], [x[0][1]for x in hyps]
		yMean, xMean = statistics.mean(yVals), statistics.mean(xVals)
		yDev, xDev = statistics.pstdev(yVals) * m, statistics.pstdev(xVals) * m
		hypDict[key] = [x for x in hyps if not determineOutlier(x[0], yMean, yDev, xMean, xDev)]
	
def determineOutlier(input, yMean, yDev, xMean, xDev):
	return abs(input[0] - yMean) > yDev or abs(input[1] - xMean) > xDev

def dictToArray(hypDict):
	coordArray = np.zeros((len(hypDict.keys()), 2))
	for key, hyps in hypDict.items():
		coordArray[key] = np.array([round(hyps[1]), round(hyps[0])]) # x, y format
	return coordArray
	
def getMean(hypDict): # get weighted average of coordinates, weights list
	meanDict = {}
	for key, hyps in hypDict.items():
		xMean = 0
		yMean = 0
		totalWeight = 0
		for hyp in hyps:
			yMean += hyp[0][0] * hyp[1]
			xMean += hyp[0][1] * hyp[1]
			totalWeight += hyp[1]
		yMean /= totalWeight
		xMean /= totalWeight
		meanDict[key] = [yMean, xMean]
	return meanDict
	
def getCovariance(hypDict, meanDict):
	covarDict = {}
	for key, hyps in hypDict.items():
		numerator = 0
		totalWeight = 0
		for hyp in hyps:
			yDiff = hyp[0][0] - meanDict[key][0]
			xDiff = hyp[0][1] - meanDict[key][0]
			numerator += (yDiff ** 2 + xDiff ** 2) * hyp[1]
			totalWeight += hyp[1]
		covarDict[key] = numerator / totalWeight
	return covarDict
	
def getMeanArray(inDict):
	meanDict = {}
	for key, hyps in inDict.items():
		entry = np.zeros(2)
		totalWeight = 0
		for hyp in hyps:
			entry += np.array(hyp[0]) * hyp[1]
			totalWeight += hyp[1]
		entry /= totalWeight
		meanDict[key] = entry
	return meanDict
	
def displayMaskChoice(inArray): # used for mask output of shape (height, width, 2)
	outArray = [ [1 if x[1] >= x[0] else 0 for x in y] for y in inArray]
	showArrayAsImage(np.array(outArray))

def showImage(img): # displays image using plt
	plt.figure()
	imshow(np.squeeze(img))
	plt.show()

def testModelMask(modelName, modelStruct, tests = 5, modelClass = 'cat', outClasses = False, outVectors = False, optimizer = tf.keras.optimizers.Adam, learning_rate = 0.01, losses = None, metrics = ['accuracy']): # tests mask model output
	if not (outVectors or outClasses):
		print("At least one of outVectors or outClasses must be set to True.")
		return
	
	model = modelStruct(outVectors = outVectors, outClasses = outClasses)
	model.load_weights(os.path.dirname(os.path.realpath(__file__)) + '\\models\\' + modelName + '_' + modelClass)
	model.summary()
	
	#model = tf.keras.models.load_model(modelStruct, modelName, modelClass = modelClass, outVectors = outVectors, outClasses = outClasses, optimizer = optimizer, learning_rate = learning_rate, losses = losses, metrics = metrics)
	
	basePath = os.path.dirname(os.path.realpath(__file__)) + '\\LINEMOD\\' + modelClass
	plt.figure()
	for i in range(tests):
		randNum = random.randrange(len(os.listdir(basePath + '\\JPEGImages\\')))
		orig = imread(basePath + '\\JPEGImages\\' + os.listdir(basePath + '\\JPEGImages\\')[randNum])
		#orig2 = data.filePathToArray(basePath + '\\JPEGImages\\' + os.listdir(basePath + '\\JPEGImages\\')[randNum])
		pred = model.predict(np.array([orig]))
		
		plt.subplot(211)
		imshow(orig)
		
		plt.subplot(212)
		imshow(np.squeeze(pred))
		plt.show()

def pnp(p3d, p2d, drawPoints, matrix = np.array([[572.4114, 0., 325.2611], [0., 573.57043, 242.04899], [0., 0., 1.]]), method = cv2.SOLVEPNP_ITERATIVE):

	assert p3d.shape[0] == p2d.shape[0], 'points 3D and points 2D must have same number of vertices'

	p2d = np.ascontiguousarray(p2d.astype(np.float64))
	p3d = np.ascontiguousarray(p3d.astype(np.float64))
	matrix = matrix.astype(np.float64)
	try:
		_, R_exp, tVec = cv2.solvePnP(p3d,
								p2d,
								matrix,
								np.zeros(shape=[8, 1], dtype='float64'),
								flags=method)
	except Exception as e:
		print(e)
		set_trace()
		print(p2d)

	#R_exp, t, _ = cv2.solvePnPRansac(p3d,
	#							p2d,
	#							matrix,
	#							distCoeffs,
	#							reprojectionError=12.0)

	#R, _ = cv2.Rodrigues(R_exp)
	
	(plotPoints, jacobian) = cv2.projectPoints(drawPoints, R_exp, tVec, matrix, np.zeros(shape=[8, 1], dtype='float64'))
	
	# return np.concatenate([R, tVec], axis=-1)
	return np.squeeze(plotPoints)

def drawPose(img, drawPoints, colour = (255,0,0)): # draw bounding box
	
	cv2.line(img, drawPoints['bld'], drawPoints['blu'], colour, 2)
	cv2.line(img, drawPoints['bld'], drawPoints['fld'], colour, 2)
	cv2.line(img, drawPoints['bld'], drawPoints['brd'], colour, 2)
	cv2.line(img, drawPoints['blu'], drawPoints['flu'], colour, 2)
	cv2.line(img, drawPoints['blu'], drawPoints['bru'], colour, 2)
	cv2.line(img, drawPoints['fld'], drawPoints['flu'], colour, 2)
	cv2.line(img, drawPoints['fld'], drawPoints['frd'], colour, 2)
	cv2.line(img, drawPoints['flu'], drawPoints['fru'], colour, 2)
	cv2.line(img, drawPoints['fru'], drawPoints['bru'], colour, 2)
	cv2.line(img, drawPoints['fru'], drawPoints['frd'], colour, 2)
	cv2.line(img, drawPoints['frd'], drawPoints['brd'], colour, 2)
	cv2.line(img, drawPoints['brd'], drawPoints['bru'], colour, 2)
	
def accuracyPlot(modelSets, removeOutliers = False, m = 2):
	modelNames, mses, maes = [], [], []
	for modelSet in modelSets:
		if type(modelSet.name) == dict:
			modelName = modelSet.name['classModel'] + '_x_' + modelSet.name['vecModel']
		else:
			modelName = modelSet.name
		modelNames.append(modelName)
		
		with open("accuracyHistory\\" + modelName + '_' + modelSet.modelClass, 'rb') as f: # loading old history 
			predResults = pickle.load(f)
			
		if removeOutliers:
			mseSet = tf.keras.metrics.mean_squared_error(predResults['true'], predResults['pred'])
			maeSet = tf.keras.metrics.mean_absolute_error(predResults['true'], predResults['pred'])
			mseList, maeList = [], []
			for i in range(maeSet.shape[0]):
				mseList.append(tf.reduce_mean(mseSet[i]).numpy())
				maeList.append(tf.reduce_mean(maeSet[i]).numpy())
			
			maeStd = np.std(maeList) * m
			maeMean = np.mean(maeList)
			toRemove = []
			
			for i, el in enumerate(maeList):
				if abs(el - maeMean) > maeStd:
					toRemove.append(i)
					
			print(maeList)
			print("Pruning {0} of {1} entries. ({2}% outlier rate)".format(len(toRemove), len(maeList), round(len(toRemove) / len(maeList) * 100, 2)))
			
			for i in sorted(toRemove, reverse = True):
				print("Removed {0}".format(maeList[i]))
				del mseList[i]
				del maeList[i]
			
			mses.append(tf.reduce_mean(mseList))
			maes.append(tf.reduce_mean(maeList))
		else:
			mses.append(tf.reduce_mean(tf.keras.metrics.mean_squared_error(predResults['true'], predResults['pred'])))
			maes.append(tf.reduce_mean(tf.keras.metrics.mean_absolute_error(predResults['true'], predResults['pred'])))
		
	yPos = np.arange(len(modelNames))
	
	plt.subplot(211)
	plt.bar(yPos, maes)
	plt.xticks(yPos, modelNames)
	plt.ylabel("Mean Absolute Error")
	plt.subplot(212)
	plt.bar(yPos, mses)
	plt.xticks(yPos, modelNames)
	plt.ylabel("Mean Squared Error")
	plt.show()
	
def labelDrawPoints(drawList): # (b, f = back, front), (l, r = left, right), (u, d = up , down)
	drawDict = {}
	drawDict['bld'] = (int(round(drawList[0][0])), int(round(drawList[0][1])))
	drawDict['blu'] = (int(round(drawList[1][0])), int(round(drawList[1][1])))
	drawDict['fld'] = (int(round(drawList[2][0])), int(round(drawList[2][1])))
	drawDict['flu'] = (int(round(drawList[3][0])), int(round(drawList[3][1])))
	drawDict['brd'] = (int(round(drawList[4][0])), int(round(drawList[4][1])))
	drawDict['bru'] = (int(round(drawList[5][0])), int(round(drawList[5][1])))
	drawDict['frd'] = (int(round(drawList[6][0])), int(round(drawList[6][1])))
	drawDict['fru'] = (int(round(drawList[7][0])), int(round(drawList[7][1])))
	return drawDict

def evalModels(modelSets, trials = 5, showImageChoice = False, showTrue = False, saveImage = False, saveAccuracy = False, allValid = False):
	for modelSet in modelSets:
		if type(modelSet.name) == dict:
			if len(modelSet.name) == 2: # seperate models
				classModel = models.modelsDict[modelSet.name['classModel']]
				classModel = models.loadModelWeights(classModel.structure, modelSet.name['classModel'], modelClass = modelSet.modelClass, outVectors = False, outClasses = True, losses = classModel.losses)
				vecModel = models.modelsDict[modelSet.name['vecModel']]
				altLab = vecModel.altLabels
				vecModel = models.loadModelWeights(vecModel.structure, modelSet.name['vecModel'], modelClass = modelSet.modelClass, outVectors = True, outClasses = False, losses = vecModel.losses)
				modelWrap = modelWrapper({'classModel': classModel, 'vecModel': vecModel}, altLabels = altLab)
				modelName = modelSet.name['classModel'] + '_x_' + modelSet.name['vecModel']
			else:
				raise Exception("Probably shouldn't be here ever..")
		else: # combined model
			fullModel = models.modelsDict[modelSet.name]
			modelWrap = modelWrapper(models.loadModelWeights(fullModel.structure, modelSet.name, modelClass = modelSet.modelClass, outVectors = True, outClasses = True, losses = fullModel.losses), altLabels = fullModel.altLabels)
			modelName = modelSet.name
			
		print("Evaluating {0}:".format(str(modelSet.name)))
		
		trueList, predList = [], []
		
		if allValid:
			validData = data.getDataSplit(modelClass = modelSet.modelClass)[1]
			basePath = os.path.dirname(os.path.realpath(__file__)) + '\\LINEMOD\\' + modelSet.modelClass
			trials = len(validData)
		
		for i in range(trials):
		
			if allValid:
				with open(basePath + '\\labels\\' + validData[i][2]) as f:
					labels = f.readline().split(' ')[1:19]
				image = data.filePathToArray(basePath + '\\JPEGImages\\' + validData[i][0])
			else:
				image, labels = data.getDataSplitImage(True)
			
			coordsPred, classPred = modelWrap.genPredict(np.array([image]))

			#yPred = predictPose(coordsPred[0], classPred[0], labels = labels, addNoise = addNoise, checkPreds = image)
			res, yPred = predictPose(coordsPred[0], classPred[0], altLabels = modelWrap.altLabels, pruning = pruneBool)
		
			if not res:
				print("Couldn't find it..")
				continue
			else:
				yTrue = data.labelFloatsToPixels(labels, decPlace = 8)[1::]
				
				#print("True: \n{0}\nPred:\n{1}".format(yTrue, yPred))
				
				trueList.append(yTrue)
				predList.append(yPred)
				print("Prediction {0} of {1} - MAE: {2}, MSE {3}".format(i + 1, trials, round(tf.reduce_mean(tf.keras.metrics.mean_absolute_error(yTrue, yPred)).numpy(), 2), round(tf.reduce_mean(tf.keras.metrics.mean_squared_error(yTrue, yPred)).numpy(), 2)))
				
				if showImageChoice:
					img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
					drawPose(img, labelDrawPoints(yPred))
					if showTrue:
						drawPose(img, labelDrawPoints(yTrue), (0,255,0))
					cv2.imshow("Output", img)
					cv2.waitKey()
				
				if saveImage:
					img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
					drawPose(img, labelDrawPoints(yPred))
					drawPose(img, labelDrawPoints(yTrue), (0,255,0))
					cv2.imwrite("savedImages\\{0} - {1}.jpg".format(modelName, i), img)
		
		trueList = np.array(trueList)
		predList = np.array(predList)
		print("Avg MAE: {0}, Avg MSE {1}".format(tf.reduce_mean(tf.keras.metrics.mean_absolute_error(trueList, predList)), tf.reduce_mean(tf.keras.metrics.mean_squared_error(trueList, predList))))
		
		if saveAccuracy:
			#with open("accuracyHistory\\{0}_{1}_{2}_trials_{3}_hyps{4}".format(modelName, modelSet.modelClass, trials, numHypotheses, ('_pruning{0}'.format(pruneRatio) if pruneBool else '')), 'wb') as f: # create model history
			with open("accuracyHistory\\" + modelName + '_' + modelSet.modelClass, 'wb') as f: # create model history
				pickle.dump({'true': trueList, 'pred': predList}, f)

if __name__ == "__main__" :
	modelSets = [modelSet({'classModel': 'uNet_classes', 'vecModel': 'stvNet_new_coords_alt'}), modelSet({'classModel': 'uNet_classes', 'vecModel': 'stvNet_new_coords'})]
	#modelSets = [modelSet('stvNetAltLabels'), modelSet('stvNetNormLabels')]
	#evalModels(modelSets, allValid = True)
	#modelSets = [modelSet({'classModel': 'uNet_classes', 'vecModel': 'stvNet_new_coords_alt'})]
	evalModels(modelSets, trials = 10, showImageChoice = False, showTrue = True, saveImage = False, saveAccuracy = False, allValid = False)
	#accuracyPlot(modelSets, True)