import os, numpy as np, math, random, matplotlib.pyplot as plt, pickle, cv2
from PIL import Image

def augmentData(x, modelMask, labels): # for data aug, get random horizontal, vertical flips, flip input x with np, label vals = 1 - labelvals, flip mask
	if random.choice([True, False]): # vertical flip
		x = np.flipud(x)
		modelMask = np.flipud(modelMask)
		for i in range(len(labels)):
			labels[i][1] = round(1 - float(labels[i][1]), 6)
	if random.choice([True, False]): #  horizontal flip
		x = np.fliplr(x)
		modelMask = np.fliplr(modelMask)
		for i in range(len(labels)):
			labels[i][0] = round(1 - float(labels[i][0]), 6)
	
	return x , modelMask, labels

def getDataSplitImage(getValid, modelClass = 'cat', bb8Labels = True): # retrieves random image and label set from specified dataset
	trainData, validData = getDataSplit(modelClass = modelClass)
	basePath = os.path.dirname(os.path.realpath(__file__)) + '\\LINEMOD\\' + modelClass
	choice = random.choice(validData if getValid else trainData)
	keypoints = np.loadtxt(basePath + f'\\{"bb8" if bb8Labels else "fps"}Labels\\' + choice[2], delimiter=',')
	image = filePathToArray(basePath + '\\JPEGImages\\' + choice[0])
	return image, keypoints

def getMasterList(basePath): # returns list with image, mask, and label filenames
	imageList = os.listdir(basePath + '\\JPEGImages\\')
	maskList = os.listdir(basePath + '\\mask\\')
	labelList = os.listdir(basePath + f'\\bb8Labels\\')
	if len(imageList) != len(maskList) or len(imageList) != len(labelList):
		raise Exception("image, mask, and label list lengths do not match.")
	
	return [[a, b, c] for a, b, c in zip(imageList, maskList, labelList)]

def classTrainingGenerator(model, batchSize, masterList = None, height = 480, width = 640, augmentation = True): # take input image, resize and store as rgb, create mask training data
	basePath = os.path.dirname(os.path.realpath(__file__)) + '\\LINEMOD\\' + model
	if masterList == None:
		masterList = getMasterList(basePath)
		random.shuffle(masterList)
	i = 0
	while True:
		xBatch = []
		yClassBatch = []
		for _ in range(batchSize):
			if i == len(masterList):
				i = 0
				random.shuffle(masterList)
			x = filePathToArray(basePath + '\\JPEGImages\\' + masterList[i][0], height, width)
			
			yClassLabels = np.zeros((height, width, 1)) # 1 class confidence value per model
			modelMask = filePathToArray(basePath + '\\mask\\' + masterList[i][1], height, width)
			
			if augmentation:
				if random.choice([True, False]): # vertical flip
					x = np.flipud(x)
					modelMask = np.flipud(modelMask)
				if random.choice([True, False]): #  horizontal flip
					x = np.fliplr(x)
					modelMask = np.fliplr(modelMask)
			
			modelCoords = np.where(modelMask == 255)[:2]
			
			for modelCoord in zip(modelCoords[0][::3], modelCoords[1][::3]):
				yClassLabels[modelCoord[0]][modelCoord[1]][0] = 1
			
			xBatch.append(x)
			yClassBatch.append(yClassLabels)
			i += 1
		#print(np.array(yClassBatch).shape)
		yield (np.array(xBatch), np.array(yClassBatch))

def coordsTrainingGenerator(model, batchSize, masterList = None, height = 480, width = 640, augmentation = True, bb8Labels = True): # takes input image and generates unit vector training data
	basePath = os.path.dirname(os.path.realpath(__file__)) + '\\LINEMOD\\' + model
	if masterList == None:
		masterList = getMasterList(basePath)
		random.shuffle(masterList)
	i = 0
	while True:
		xBatch = []
		yCoordBatch = []
		for _ in range(batchSize):
			if i == len(masterList):
				i = 0
				random.shuffle(masterList)
			x = filePathToArray(basePath + '\\JPEGImages\\' + masterList[i][0], height, width)

			labels = np.loadtxt(basePath + f'\\{"bb8" if bb8Labels else "fps"}Labels\\' + masterList[i][2], delimiter=',')
			
			yCoordsLabels = np.zeros((height, width, 18)) # 9 coordinates
			
			modelMask = filePathToArray(basePath + '\\mask\\' + masterList[i][1], height, width)
			
			if augmentation: 
				x, modelMask, labels = augmentData(x, modelMask, labels)
			
			modelCoords = np.where(modelMask == 255)[:2]
			for modelCoord in zip(modelCoords[0][::3], modelCoords[1][::3]):
				setTrainingPixel(yCoordsLabels, modelCoord[0], modelCoord[1], labels, height, width)
			xBatch.append(x)
			yCoordBatch.append(yCoordsLabels)
			i += 1
		yield (np.array(xBatch), np.array(yCoordBatch))

def combinedTrainingGenerator(model, batchSize, masterList = None, height = 480, width = 640, out0 = 'activation_9', out1 = 'activation_10', augmentation = True, bb8Labels = True): # take input image, resize and store as rgb, create training data
	basePath = os.path.dirname(os.path.realpath(__file__)) + '\\LINEMOD\\' + model
	if masterList == None:
		masterList = getMasterList(basePath)
	i = 0
	while True:
		xBatch = []
		yCoordBatch = []
		yClassBatch = []
		for _ in range(batchSize):
			if i == len(masterList):
				i = 0
				random.shuffle(masterList)
			x = filePathToArray(basePath + '\\JPEGImages\\' + masterList[i][0], height, width)
			
			labels = np.loadtxt(basePath + f'\\{"bb8" if bb8Labels else "fps"}Labels\\' + masterList[i][2], delimiter=',')
			
			yCoordsLabels = np.zeros((height, width, 18)) # 9 coordinates
			yClassLabels = np.zeros((height, width, 1)) # 1 class confidence value per model
			
			modelMask = filePathToArray(basePath + '\\mask\\' + masterList[i][1], height, width)
			
			if augmentation: 
				x, modelMask, labels = augmentData(x, modelMask, labels)
			
			modelCoords = np.where(modelMask == 255)[:2]
			for modelCoord in zip(modelCoords[0][::3], modelCoords[1][::3]):
				setTrainingPixel(yCoordsLabels, modelCoord[0], modelCoord[1], labels, height, width)
				yClassLabels[modelCoord[0]][modelCoord[1]][0] = 1
			xBatch.append(x)
			yCoordBatch.append(yCoordsLabels)
			yClassBatch.append(yClassLabels)
			i += 1
		yield (np.array(xBatch), {out0: np.array(yCoordBatch), out1 : np.array(yClassBatch)})

def filePathToArray(filePath, height = 480, width = 640): # uses PIL Image object to return image as numpy array
	image = Image.open(filePath)
	image = image.resize((width, height))
	return np.array(image)

def showArrayAsImage(inArray, scaler = 255, mode = 'F', saveImage = False): # displays image using PIL Image object
	displayImage = inArray * scaler
	displayImage = Image.fromarray(np.squeeze(displayImage), mode)
	displayImage.show()
	if saveImage:
		displayImage = displayImage.convert("L")
		displayImage.save("maskOutput.png", "png")

def setTrainingPixel(outImage, y, x, labels, height, width): # for each pixel given, calculate unit vectors to keypoints and store on pixel in outImage object
	for i in range(len(labels)):
		yDiff = height * float(labels[i][1]) - y # positive means y is above target in image
		xDiff = width * float(labels[i][0]) - x # positive means x is left of target in image
		mag = math.sqrt(yDiff ** 2 + xDiff ** 2)
		
		outImage[y][x][i * 2 + 1] = yDiff / mag # assign unit vectors pointing from coordinate to keypoint
		outImage[y][x][i * 2] = xDiff / mag

def showKeypoints(model = 'cat', batchSize = 2, height = 480, width = 640): # display labelled keypoints on image
	basePath = os.path.dirname(os.path.realpath(__file__)) + '\\LINEMOD\\' + model
	masterList = getMasterList(basePath)
	i = 0
	for _ in range(batchSize):
		if i == len(masterList):
			i = 0
			random.shuffle(masterList)
		print(masterList[i][0])
		x = filePathToArray(basePath + '\\JPEGImages\\' + masterList[i][0], height, width)
		
		with open(basePath + '\\labels\\' + masterList[i][2]) as f:
			labels = f.readline().split(' ')[1:19]
		
		for ind in range(len(labels) // 2):
			px = round(float(labels[ind * 2]) * width)
			py = round(float(labels[ind * 2 + 1]) * height)
			print("keypoint at " + str((px, py)))
			temp = np.array(x[py][px])
			x[py][px] = np.array([0,0,0])
			plt.figure()
			plt.imshow(np.squeeze(x))
			plt.show()
			x[py][px] = temp
		i += 1

def labelFloatsToPixels(floatList, height = 480, width = 640, decPlace = 0): # takes normalized pixel labels, converts to integer coordinates
	labelList = []
	
	for coord in floatList:
		labelList.append([round(float(coord[0]) * width, decPlace), round(float(coord[1]) * height, decPlace)]) # x, y format
		
	return labelList

def getDataSplit(genNew = False, split = 0.8, modelClass = 'cat'): # access training data, get jpeg, mask, label filenames split into training / validation sets
	if genNew: # create split
		basePath = os.path.dirname(os.path.realpath(__file__)) + '\\LINEMOD\\' + modelClass
		masterList = getMasterList(basePath)
		random.shuffle(masterList)
		
		splitPoint = round(len(masterList) * split)
		
		splitDict = {}
		
		splitDict["trainData"] = masterList[:splitPoint]
		splitDict["validData"] = masterList[splitPoint:]
		
		with open(f"{modelClass}_trainSplit", 'wb') as f:
			pickle.dump(splitDict, f)
		
	else: # load saved split
		with open(f"{modelClass}_trainSplit", 'rb') as f:
			splitDict = pickle.load(f)
	return (splitDict["trainData"], splitDict["validData"])

def genAltLabels(p3dOld, p3dNew, matrix = np.array([[572.4114, 0., 325.2611], [0., 573.57043, 242.04899], [0., 0., 1.]]), method = cv2.SOLVEPNP_ITERATIVE, modelClass = 'cat', height = 480, width = 640, showPoint = False): # generate pixel labels for p3dNew using labels for p3dOld
	
	p3dOld = np.ascontiguousarray(p3dOld.astype(np.float64))
	p3dOld = np.append([[0, 0, 0]], p3dOld, 0)
	
	labelDict = {'ape': 0, 'benchvise': 1, 'cam': 2, 'can': 3, 'cat': 4, 'driller': 5, 'duck': 6, 'eggbox': 7, 'glue': 8, 'holepuncher': 9, 'iron': 10, 'lamp': 11, 'phone': 12}
	basePath = os.path.dirname(os.path.realpath(__file__)) + '\\LINEMOD\\' + modelClass
	masterList = getMasterList(basePath)
	
	labelPath = basePath + '\\fpsLabels\\'
	newLabelPath = basePath + '\\altLabels\\'
	for el in masterList:
		labels = np.loadtxt(labelPath + el[2], delimiter=',')
		
		labels = np.array([[el[0] * width, el[1] * height] for el in labels])
		
		p2d = np.ascontiguousarray(labels.astype(np.float64))
		
		_, R_exp, tVec = cv2.solvePnP(p3dOld, p2d, matrix, np.zeros(shape=[8, 1], dtype='float64'), flags=method)
		
		(plotPoints, jacobian) = cv2.projectPoints(p3dNew, R_exp, tVec, matrix, np.zeros(shape=[8, 1], dtype='float64'))
		
		print(plotPoints)
		
		image = filePathToArray(basePath + '\\JPEGImages\\' + el[0])
		
		#print("looking at {0}".format(el[0]))
		
		newLabels = [labelDict[modelClass]]
		for coord in plotPoints:
			if showPoint:
				px = int(round(coord[0][0]))
				py = int(round(coord[0][1]))
				print("keypoint at " + str((px, py)))
				temp = np.array(image[py][px])
				image[py][px] = np.array([0,0,0])
				plt.figure()
				plt.imshow(np.squeeze(image))
				plt.show()
				image[py][px] = temp
			newLabels.append(coord[0][0] / width)
			newLabels.append(coord[0][1] / height)
			
		with open(newLabelPath + el[2], 'w') as f:
			for lab in newLabels:
				f.write(str(lab) + ' ')