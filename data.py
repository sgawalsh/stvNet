import os, pandas as pd, numpy as np, tensorflow as tf, math, random, matplotlib.pyplot as plt, sys, pickle, cv2
from PIL import Image
from pdb import set_trace
from matplotlib import pyplot
from decimal import Decimal
from skimage.io import imshow

def getRandomImage(modelClass = 'cat'): # retrieves random image and label set
	basePath = os.path.dirname(os.path.realpath(__file__)) + '\\LINEMOD\\' + modelClass
	imageList = os.listdir(basePath + '\\JPEGImages\\')
	randNum = random.randrange(len(imageList))
	image = imageList[randNum]
	with open(basePath + '\\labels\\' + os.listdir(basePath + '\\labels\\')[randNum]) as f:
		labels = f.readline().split(' ')[1:19]
	print(image)
	image = filePathToArray(basePath + '\\JPEGImages\\' + image)
	#image = imread(basePath + image)
	return image, labels

def getDataSplitImage(getValid, modelClass = 'cat'): # retrieves random image and label set from specified dataset
	trainData, validData = getDataSplit(modelClass = modelClass)
	basePath = os.path.dirname(os.path.realpath(__file__)) + '\\LINEMOD\\' + modelClass
	if getValid:
		choice = random.choice(validData)
		with open(basePath + '\\labels\\' + choice[2]) as f:
			labels = f.readline().split(' ')[1:19]
		image = filePathToArray(basePath + '\\JPEGImages\\' + choice[0])
	else:
		choice = random.choice(trainData)
		with open(basePath + '\\labels\\' + choice[2]) as f:
			labels = f.readline().split(' ')[1:19]
		image = filePathToArray(basePath + '\\JPEGImages\\' + choice[0])
	return image, labels

def getMasterList(basePath): # returns list with image, mask, and label filenames
	imageList = os.listdir(basePath + '\\JPEGImages\\')
	maskList = os.listdir(basePath + '\\mask\\')
	labelList = os.listdir(basePath + '\\labels\\')
	if len(imageList) != len(maskList) or len(imageList) != len(labelList):
		raise Exception("image, mask, and label list lengths do not match.")
	
	return [[a, b, c] for a, b, c in zip(imageList, maskList, labelList)]

def classTrainingGenerator(model, batchSize, masterList = None, height = 480, width = 640, augmentation = True, **unused): # take input image, resize and store as rgb, create mask training data
	basePath = os.path.dirname(os.path.realpath(__file__)) + '\\LINEMOD\\' + model
	if masterList == None:
		masterList = getMasterList(basePath)
		random.shuffle(masterList)
	i = 0
	while True:
		xBatch = []
		yClassBatch = []
		for b in range(batchSize):
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

def coordsTrainingGenerator(model, batchSize, masterList = None, height = 480, width = 640, augmentation = True, altLabels = True): # takes input image and generates unit vector training data
	basePath = os.path.dirname(os.path.realpath(__file__)) + '\\LINEMOD\\' + model
	if masterList == None:
		masterList = getMasterList(basePath)
		random.shuffle(masterList)
	i = 0
	while True:
		xBatch = []
		yCoordBatch = []
		for b in range(batchSize):
			if i == len(masterList):
				i = 0
				random.shuffle(masterList)
			x = filePathToArray(basePath + '\\JPEGImages\\' + masterList[i][0], height, width)
			
			with open(basePath + ('\\altLabels\\' if altLabels else '\\labels\\') + masterList[i][2]) as f:
				labels = f.readline().split(' ')[1:19]
			
			yCoordsLabels = np.zeros((height, width, 18)) # 9 coordinates
			
			modelMask = filePathToArray(basePath + '\\mask\\' + masterList[i][1], height, width)
			
			if augmentation:
				if random.choice([True, False]): # vertical flip
					x = np.flipud(x)
					modelMask = np.flipud(modelMask)
					for i in range(len(labels) // 2):
						labels[i * 2 + 1] = str(round(1 - float(labels[i * 2 + 1]), 6))
				if random.choice([True, False]): #  horizontal flip
					x = np.fliplr(x)
					modelMask = np.fliplr(modelMask)
					for i in range(len(labels) // 2):
						labels[i * 2] = str(round(1 - float(labels[i * 2]), 6))
			
			modelCoords = np.where(modelMask == 255)[:2]
			for modelCoord in zip(modelCoords[0][::3], modelCoords[1][::3]):
				setTrainingPixel(yCoordsLabels, modelCoord[0], modelCoord[1], labels, height, width)
			xBatch.append(x)
			yCoordBatch.append(yCoordsLabels)
			i += 1
		yield (np.array(xBatch), np.array(yCoordBatch))


def combinedTrainingGenerator(model, batchSize, masterList = None, height = 480, width = 640, out0 = 'activation_9', out1 = 'activation_10', augmentation = True, altLabels = True): # take input image, resize and store as rgb, create training data
	basePath = os.path.dirname(os.path.realpath(__file__)) + '\\LINEMOD\\' + model
	if masterList == None:
		masterList = getMasterList(basePath)
	i = 0
	while True:
		xBatch = []
		yCoordBatch = []
		yClassBatch = []
		for b in range(batchSize):
			if i == len(masterList):
				i = 0
				random.shuffle(masterList)
			x = filePathToArray(basePath + '\\JPEGImages\\' + masterList[i][0], height, width)
			
			with open(basePath + ('\\altLabels\\' if altLabels else '\\labels\\') + masterList[i][2]) as f:
				labels = f.readline().split(' ')[1:19]
			
			yCoordsLabels = np.zeros((height, width, 18)) # 9 coordinates
			yClassLabels = np.zeros((height, width, 1)) # 1 class confidence value per model
			#yClassLabels = np.tile(np.array([1, 0]),(height, width, 1))
			
			modelMask = filePathToArray(basePath + '\\mask\\' + masterList[i][1], height, width)
			
			if augmentation: # for data aug, get random horizontal, vertical flips, flip input x with np, label vals = 1 - labelvals, flip mask
				if random.choice([True, False]): # vertical flip
					x = np.flipud(x)
					modelMask = np.flipud(modelMask)
					for i in range(len(labels) // 2):
						labels[i * 2 + 1] = str(round(1 - float(labels[i * 2 + 1]), 6))
				if random.choice([True, False]): #  horizontal flip
					x = np.fliplr(x)
					modelMask = np.fliplr(modelMask)
					for i in range(len(labels) // 2):
						labels[i * 2] = str(round(1 - float(labels[i * 2]), 6))
			
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
	for i in range(9):
		yDiff = height * float(labels[i * 2 + 1]) - y # positive means y is above target in image
		xDiff = width * float(labels[i * 2]) - x # positive means x is left of target in image
		mag = math.sqrt(yDiff ** 2 + xDiff ** 2)
		
		outImage[y][x][i * 2 + 1] = yDiff / mag # assign unit vectors pointing from coordinate to keypoint
		outImage[y][x][i * 2] = xDiff / mag

def showKeypoints(model = 'cat', batchSize = 2, height = 480, width = 640): # display labelled keypoints on image
	basePath = os.path.dirname(os.path.realpath(__file__)) + '\\LINEMOD\\' + model
	masterList = getMasterList(basePath)
	i = 0
	for b in range(batchSize):
		if i == len(masterList):
			i = 0
			random.shuffle(masterList)
		print(masterList[i][0])
		x = filePathToArray(basePath + '\\JPEGImages\\' + masterList[i][0], height, width)
		
		with open(basePath + '\\labels\\' + masterList[i][2]) as f:
			labels = f.readline().split(' ')[1:19]
		
		yCoordsLabels = np.zeros((height, width, 18)) # 9 coordinates
		
		for ind in range(len(labels) // 2):
			px = round(float(labels[ind * 2]) * width)
			py = round(float(labels[ind * 2 + 1]) * height)
			print("keypoint at " + str((px, py)))
			temp = np.array(x[py][px])
			x[py][px] = np.array([0,0,0])
			plt.figure()
			imshow(np.squeeze(x))
			plt.show()
			x[py][px] = temp
		i += 1

def labelFloatsToPixels(floatList, height = 480, width = 640, decPlace = 0): # takes normalized pixel labels, converts to integer coordinates
	labelList = []
	
	for ind in range(len(floatList) // 2):
		labelList.append([round(float(floatList[ind * 2]) * width, decPlace), round(float(floatList[ind * 2 + 1]) * height, decPlace)]) # x, y format
		
	return labelList

def getDataSplit(genNew = False, split = .8, modelClass = 'cat'): # access training data, get jpeg, mask, label filenames split into training / validation sets
	if genNew: # create split
		basePath = os.path.dirname(os.path.realpath(__file__)) + '\\LINEMOD\\' + modelClass
		masterList = getMasterList(basePath)
		random.shuffle(masterList)
		
		splitPoint = round(len(masterList) * .8)
		
		splitDict = {}
		
		splitDict["trainData"] = masterList[:splitPoint]
		splitDict["validData"] = masterList[splitPoint:]
		
		with open("{0}_trainSplit".format(modelClass), 'wb') as f:
			pickle.dump(splitDict, f)
		
	else: # load saved split
		with open("{0}_trainSplit".format(modelClass), 'rb') as f:
			splitDict = pickle.load(f)
	return (splitDict["trainData"], splitDict["validData"])

def genAltLabels(p3dOld, p3dNew, matrix = np.array([[572.4114, 0., 325.2611], [0., 573.57043, 242.04899], [0., 0., 1.]]), method = cv2.SOLVEPNP_ITERATIVE, modelClass = 'cat', height = 480, width = 640, showPoint = False): # generate pixel labels for p3dNew using labels for p3dOld
	
	p3dOld = np.ascontiguousarray(p3dOld.astype(np.float64))
	p3dOld = np.append([[0, 0, 0]], p3dOld, 0)
	
	labelDict = {'ape': 0, 'benchvise': 1, 'cam': 2, 'can': 3, 'cat': 4, 'driller': 5, 'duck': 6, 'eggbox': 7, 'glue': 8, 'holepuncher': 9, 'iron': 10, 'lamp': 11, 'phone': 12}
	basePath = os.path.dirname(os.path.realpath(__file__)) + '\\LINEMOD\\' + modelClass
	masterList = getMasterList(basePath)
	
	labelPath = basePath + '\\labels\\'
	newLabelPath = basePath + '\\altLabels\\'
	for el in masterList:
		with open(labelPath + el[2], 'r') as f:
			labels = f.readline().split(' ')[1:19] # ignore class label and centroid
		
		labels = [float(el) for el in labels]
		labels = np.reshape(labels, (p3dOld.shape[0], 2))
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
				imshow(np.squeeze(image))
				plt.show()
				image[py][px] = temp
			newLabels.append(coord[0][0] / width)
			newLabels.append(coord[0][1] / height)
			
		with open(newLabelPath + el[2], 'w') as f:
			for lab in newLabels:
				f.write(str(lab) + ' ')