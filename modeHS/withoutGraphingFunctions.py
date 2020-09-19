import numpy as np
import cv2
import matplotlib.pyplot as plt
import helperFunctions

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

mainestOfTheMain()
# mainestOfTheMain()