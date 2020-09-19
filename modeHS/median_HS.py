import numpy as np
import cv2
import math

def Psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def dataToBitArray(data):
    bitStream = "{0:b}".format(data)
    bitStream = list(map(int,list(bitStream)))
    return bitStream

def intToBitArray(data):
    bitStream = "{0:b}".format(data)
    bitStream = list(map(int,list(bitStream)))
    bitStream = [0]*(8-len(bitStream)) + bitStream
    return bitStream

def mtoa(matrix):
	return np.reshape(np.array(matrix),-1)

def med(array):
	a = list(array)
	a.sort()
	n = len(a)
	return a[n//2]

def lefnu(array, ma):
	a = list(array)
	a.sort()
	n = len(a)
	count = 0
	print(a)
	for i in range(n//2,-1,-1):
		if a[i] == ma:
			count +=1
		else:
			return count-1
	return count-1

def senderSide(originalImage, n, Data, Tn):
	medianImage = np.array(originalImage, dtype = np.uint8)
	step = 2*n+1
	l,w = medianImage.shape
	# a = np.reshape(medianImage,-1)
	bitStream = dataToBitArray(Data)
	# Adding 0011111111110000 after data bits
	bitStream += [0,0] + [1]*10 + [0,0,0,0]
	length = (l//step - 1) * step
	width = (w//step - 1) * step
	auxMedians = []
	medianIndex = []
	listForHist = []
	for i in range(0,length, step):
	    for j in range(0,width, step):
	        matrix = np.array(medianImage[i:i+step][j:j+step])
	        raster = mtoa(matrix)
	        medianElement = med(raster)
	        nu = lefnu(raster,medianElement)
	        auxMedians.append(medianElement)
	        count = 0
	        for i in range(len(raster)):
	        	if raster[i] == medianElement:
	        		if count == nu:
	        			medianIndex.append(i)
	        			break
	        		else:
	        			count += 1
	        count = 0
	        for x in range(step):
	            for y in range(step):
	            	if medianImage[i+x][j+y] == medianElement:
	            		if count == nu:
	            			continue
	            		else:
	            			count += 1
	            			medianImage[i+x][j+y] -= medianElement
	            			listForHist.append(medianImage[i+x][j+y])
	            	else:
	            		medianImage[i+x][j+y] -= medianElement
	            		listForHist.append(medianImage[i+x][j+y])
	hist = [0]*512
	for i in listForHist:
		hist[i+256] += 1
	zeroLeft = []
	zeroRight = []
	for i in range(257,512):
		if hist[i] == 0:
			zeroRight.append(i)
	for i in range(255,-1,-1):
		if hist[i] == 0:
			zeroLeft.append(i)
	filled = False
	minzr, maxzr, minlr, maxlr = zeroRight[0],zeroRight[-1],zeroLeft[-1],zeroLeft[0]
	ender = [0,0] + [1]*10 + [1,0,0,0]
	for i in range(0,length,step):
		for j in range(0,width,step):
			medianElement = auxMedians[0]
			medIn = medianIndex[0]
			auxMedians = auxMedians[1:]
			medianIndex = medianIndex[1:]
	        for x in range(step):
	        	for y in range(step):
	        		if medianElement<=2*Tn or medianElement >= 255- 2*Tn:
	        			medianImage[i+x][j+y] = originalImage[i+x][j+y]
	        			continue
	        		if (x*step + y) == medIn:
	        			continue
	        		else:
	        			if medianImage[i+x][j+y]>0:
	        				for rights in range(len(zeroRight)):
	        					if medianImage[i+x][j+y] >= zeroRight[rights]:
	        						k = rights
	        						break
	        				if medianImage[i+x][j+y] > Tn:
	        					value = medianImage[i+x][j+y] + medianElement + Tn + 1 - k
	        					if value>255:
	        						medianImage[i+x][j+y] = originalImage[i+x][j+y]
	        						bitStream += intToBitArray(i+x)
	        						bitStream += intToBitArray(j+y)
	        					else:
	        						medianImage[i+x][j+y] = value
	        				else:
	        					medianImage[i+x][j+y] += medianImage[i+x][j+y] + medianElement + bitStream[0] - k
	        					bitStream = bitStream[1:]
	        					if bitStream == []:
	        						filled = True
	        						bitStream += ender
	        						ender = [1]

	        			elif medianImage[i+x][j+y]<0:
	        				for lefts in range(len(zeroLeft)):
	        					if medianImage[i+x][j+y] <=zeroLeft[lefts]:
	        						u = lefts
	        						break
	        				if medianImage[i+x][j+y]>=-Tn:
	        					medianImage[i+x][j+y] += medianImage[i+x][j+y] + medianElement + u - bitStream[0]
	        					bitStream = bitStream[1:]
	        					if bitStream = []:
	        						filled = True
	        						bitStream += ender
	        						ender = [1]
	        				else:
	        					value = medianImage[i+x][j+y] + medianElement - Tn - 1 - u
	        					if value<0:
	        						medianImage[i+x][j+y] = originalImage[i+x][y+j]
	        						bitStream += intToBitArray(i+x)
	        						bitStream += intToBitArray(j+y)
	        					else:
	        						medianImage[i+x][j+y] = value
	        			else:
	        				if (x*step + y) < medIn:
	        					for lefts in range(len(zeroLeft)):
	        						if medianImage[i+x][j+y] <= zeroLeft[lefts]:
	        							u = i
	        							break
	        					medianImage[i+x][j+y] += medianImage[i+x][j+y] + medianElement + u - bitStream[0]
	        				else:
	        					for rights in range(len(zeroRight)):
	        						if medianImage[i+x][j+y] >= zeroRight[rights]:
	        							u = k
	        							break
	        					medianImage[i+x][j+y] += medianImage[i+x][j+y] + medianElement + bitStream[0] - k
        					bitStream = bitStream[1:]
        					if bitStream = []:
        						filled = True
        						bitStream += ender
        						ender = [1]
	return [medianImage, zeroRight, zeroLeft]

def receiverSide(medianImage, Tn, n, zeroRight, zeroLeft):
	extractedImage = np.array(medianImage, dtype = np.uint8)
	step = 2*n+1
	l,w = extractedImage.shape
	length = (l//step-1)*step
	width = (w//step-1)*step
	bitStream = ""
	for i in range(0,length,step):
		for j in range(0,width,step):
			matrix = np.array(extractedImage[i:i+step][j:j+step])
			raster = mtoa(matrix)
			medianElement = med(raster)
			nu = lefnu(raster, medianElement)
			count = 0
	        for rasterIndex in range(len(raster)):
	        	if raster[rasterIndex] == medianElement:
	        		if count == nu:
	        			medianIndex = rasterIndex
	        			break
	        		else:
	        			count += 1
			count = 0
	        for rasterIndex in range(len(raster)):
	        	if rasterIndex == medianIndex:
        			continue
        		else:
        			raster[i] -= medianElement
	        if (medianElement >=255-2*Tn) or (medianElement <=2*Tn):
	        	continue
	        for rasterIndex in range(len(raster)):
	        	val = raster[rasterIndex]
	        	if val>0:
					for rights in range(len(zeroRight)):
						if val >= zeroRight[rights]:
							k = rights
							break
					if val > 2*Tn + 1 - k:
						raster[rasterIndex] = medianElement + val - Tn - 1 + k
					else:
						bitStream += str(abs(val+k)//2)
						raster[rasterIndex] = medianElement + floor((val+k)/2)
				elif val<0:
    				for lefts in range(len(zeroLeft)):
    					if val <=zeroLeft[lefts]:
    						u = lefts
    						break
    				if (val >= -2*Tn-1+u):
    					raster[rasterIndex] = medianElement + ceil((val-u)/2)
    					b += str(abs(val-u)//2)
    				else:
    					raster[rasterIndex] = medianElement + val + Tn + 1 - u
    			else:
    				if rasterIndex < medianIndex:
    					raster[rasterIndex] = medianElement + ceil((val-u)/2)
    				else:
    					raster[rasterIndex] = medianElement + floor((val+k)/2)
	        raster = np.reshape(raster,(step,step))
	        for x in range(step):
	        	for y in range(step):
	        		extractedImage[i+x][j+y] = raster[x][y]
	endOfData = bitStream.index("0011111111110000")
	data = int(bitStream[:endOfData],2)
	endOfLocations = bitStream.index("0011111111111000")
	for x in range(endOfData+16,endOfLocations,16):
		i = int(bitStream[x:x+8],2)
		j = int(bitStream[x+8:x+16],2)
		extractedImage[i][j] = medianImage[i][j]
	print("Extracted data is: ", data)