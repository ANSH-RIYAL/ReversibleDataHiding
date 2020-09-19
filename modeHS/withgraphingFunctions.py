import numpy as np
import cv2
import matplotlib.pyplot as plt
import helperFunctions.py

img = cv2.imread("test.tiff",0)
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# All Averages
def allAverages(originalImage, n):
    originalImage = np.array(originalImage)
    medianImage = np.array(originalImage)
    meanImage = np.array(originalImage)
    modeImage = np.array(originalImage)
    step = 2*n+1
    l,w = img.shape
    length = (l//step) * step
    width = (w//step) * step
    for i in range(0,length, step):
        for j in range(0,width, step):
            numbers = []
            modeDict = {}
            meanSum = 0
            modeDict[-1] = 0
            for x in range(step):
                for y in range(step):
                    element = img[i+x][j+y]
                    numbers.append(element)
                    modeDict[element] = modeDict[element]+1 if (element in modeDict) else 1
                    meanSum += element
            
            numbers.sort()
            # print(numbers)
            # print(numbers[0])
            index = ((2*n+1)*(2*n+1)//2)
            # print(index)
            median = numbers[index]
            mode = -1
            for i in modeDict:
                if modeDict[i] > modeDict[mode]:
                    mode = i
            mean = meanSum//((2*n+1)*(2*n+1))
            print("mean is : {}, median is : {}, mode is : {}".format(mean, median, mode))
            for x in range(step):
                for y in range(step):
                    medianImage[i+x][j+y] -= median
                    meanImage[i+x][j+y] -= mean
                    modeImage[i+x][j+y] -= mode
    return [originalImage, medianImage, meanImage, modeImage]

img = np.asarray(img, dtype = np.int32)
[originalImage, medianImage, meanImage, modeImage] = allAverages(img,int(input("Enter Value of n")))
meanImage = np.reshape(meanImage, (-1,1))
plt.hist(meanImage, bins = 256, range = (meanImage.min(), meanImage.max()))
medianImage = np.reshape(medianImage, (-1,1))
plt.hist(medianImage, bins = 256)
modeImage = np.reshape(modeImage, (-1,1))
plt.hist(modeImage, bins = 256)

# Averages over image as a whole
img = cv2.imread("test.tiff",0)
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        totalHits[img[i][j]] += 1
        allValues.append(img[i][j])
        sumValues += img[i][j]
mean = sumValues/(img.shape[0]*img.shape[1])
imgMean = np.array(img)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        imgMean[i][j] -= mean
#cv2.imshow("mean",imgMean)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
minusAvg = []
for i in allValues:
    minusAvg.append(i-mean)
plt.hist(minusAvg, bins = 256)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        totalHits[img[i][j]] += 1
        allValues.append(img[i][j])
        sumValues += img[i][j]
mean = sumValues/(img.shape[0]*img.shape[1])
allValues.sort()
#cv2.imshow("",img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
median = allValues[256*128+1]
median
minusMedian = []
for i in allValues:
    minusMedian.append(i-median)
plt.hist(minusMedian, bins = 256)
