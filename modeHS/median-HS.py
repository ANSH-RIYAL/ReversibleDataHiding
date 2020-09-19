import numpy as np
import cv2
import matplotlib.pyplot as plt
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
    bitStream = [0]*(16-len(bitStream)) + bitStream
    return bitStream

def medianify(originalImage, n, Data):
    # Major Variables:
    # step is the size of the sub-division blocks (step = 2n+1)
    # i.e. non-overlapping blocks are of size step x step (i.e. (2n+1)x(2n+1))
    # bitStream is the payload bitstream consisting of data to be transfered and values of medians, zero-bin locations, etc.
    medianImage = np.array(originalImage, dtype = np.int16)
    step = 2*n+1
    l,w = medianImage.shape
    a = np.reshape(medianImage,-1)
    bitStream = dataToBitArray(Data)
    # Adding 0011111111110000 after data bits
    bitStream += [0,0] + [1]*10 + [0,0,0,0]
    length = (l//step - 1) * step
    width = (w//step - 1) * step
    auxMedians = []
    for i in range(0,length, step):
        for j in range(0,width, step):
            numbers = []
            for x in range(step):
                for y in range(step):
                    element = medianImage[i+x][j+y]
                    numbers.append(element)
            numbers.sort()
            index = ((2*n+1)*(2*n+1)//2)
            median = numbers[index]
            bitStream += intToBitArray(median)
            for x in range(step):
                for y in range(step):
                    medianImage[i+x][j+y] -= median
    # print(auxMedians)
    return (medianImage, bitStream)

def HistogramShift(coverImage, bitStream):
    # Major Variables:
    # pixelIntensities is a hash table storing the number of pixels per pixel intensity value
    # breakers are the explicitely reserved bit-sequences which are used to denote the 
    # breakpoints for various types of embedded bits (i.e. data, medians, minindexes, etc.)
    height = coverImage.shape[0]
    width = coverImage.shape[1]
    pixelIntensities, l = [0]*512, []
    values = np.reshape(coverImage,-1)
    maxbin = 0
    for i in values:
        pixelIntensities[i+256] += 1
        if pixelIntensities[i + 256]>pixelIntensities[maxbin]:
            maxbin = i + 256
    minbin = int(maxbin)
    for i in range(maxbin,0,-1):
        if pixelIntensities[i] < pixelIntensities[minbin]:
            minbin = i
    for i in range(maxbin,min(512,2* maxbin -minbin)):
        if pixelIntensities[i] <= pixelIntensities[minbin]:
            minbin = i
            break
    minBinVal = pixelIntensities[minbin]
    for i in range(len(values)):
        if values[i] == minbin:
            l.append(i)
    count = 0
    if maxbin<minbin:
        for i in range(len(values)):
            if maxbin < values[i] < minbin:
                values[i] += 1
    else:
        for i in range(len(values)):
            if minbin < values[i] + 256 < maxbin:
                count += 1
                values[i] -= 1
    breaker = [0,0] + [1]*12 + [0,0]
    breakerForMinVals = [0,0] + [1]*12 + [1,0]
    if minBinVal != 0:
        bitStream += breakerForMinVals
        for i in l:
            bitStream += intToBitArray(i)
    bitStream += breaker
    d = 1 if (minbin > maxbin) else -1
    for i in range(len(values)):
        if values[i] + 256 == maxbin:
            if bitStream[0] == 0:
                values[i] += d
            bitStream = bitStream[1:]
        if bitStream == []:
            break
    embedded = np.reshape(values,(height, width))
    return (embedded, minbin, maxbin)

def reverseHS(embeddedImage, minbinVal, maxbinVal):
    height = embeddedImage.shape[0]
    width = embeddedImage.shape[1]
    values = np.reshape(embeddedImage,-1)
    bitStream = ""
    d = 1 if (minbinVal > maxbinVal) else -1
    for i in range(len(values)):
        if values[i]+256 == maxbinVal:
            bitStream += "1"
        elif values[i]+256 == maxbinVal + d:
            values[i] =maxbinVal-256
            bitStream += "0"
    minVals = ""
    minValueIndexes = []
    startOfMinVal = bitStream.find("0011111111111110")
    # print(startOfMinVal)
    if startOfMinVal != -1:
        endOfMinVal = bitStream.find("0011111111111100")
        minVals = bitStream[startOfMinVal+16:endOfMinVal]
        bitStream = bitStream[:startOfMinVal]
        for i in range(0,len(minVals),16):
            minValueIndexes.append(int(minVals[i:i+16],2))
    endOfBitStream = bitStream.find("0011111111111100")
    bitStream = bitStream[:endOfBitStream]
    if minbinVal>maxbinVal:
        for i in range(len(values)):
            if maxbinVal+1<values[i]+256<=minbinVal:
                values[i] -= 1
    else:
        for i in range(len(values)):
            if minbinVal<=values[i]+256<maxbinVal-1:
                values[i] += 1
    for i in minValueIndexes:
        values[i] = minbinVal
    modulatedImage = np.reshape(values,(height,width))
    return (modulatedImage, bitStream)

def turnToOriginal(modulatedImage, n, bitStream):
    # also extracts Data
    breaker = bitStream.find("0011111111110000")
    data = int(bitStream[:breaker],2)
    bitStream = bitStream[breaker+16:]
    medians = []
    for i in range(0,len(bitStream),16):
        medians.append(int(bitStream[i:i+16],2))
    # print(medians)
    step = 2*n+1
    l,w = modulatedImage.shape
    length = (l//step-1) * step
    width = (w//step-1) * step
    for i in range(0,length, step):
        for j in range(0,width, step):
            referenceMedian = medians[0]
            medians = medians[1:]
            for x in range(step):
                for y in range(step):
                    modulatedImage[i+x][j+y] += referenceMedian
    return (modulatedImage,data)

def starterFunction():
    # First comes sender
    n = int(input("Enter n"))
    data = int(input("Enter data to be embedded"))
    img = cv2.imread("test.tiff",0)
    copyOfImage = np.array(img, dtype = np.uint8)
    img = np.asarray(img, dtype = np.int16)
    medianImage, bitStream = medianify(img,n,data)
    copyOfMedian = np.array(medianImage, dtype = np.uint8)
    print("\n\n----------------------------------------------------------\nMedians subtracted")
    embeddedImage, minBin, maxBin = HistogramShift(medianImage, bitStream)
    copyOfEmbeddedImage = np.array(embeddedImage, dtype = np.uint8)
    print("\n\n----------------------------------------------------------\nHistogram Shifting Done")
    # Now we send all these to the reciever
    medianifiedImage, bitStream = reverseHS(embeddedImage, minBin, maxBin)
    print("\n\n----------------------------------------------------------\nReverse HS done")
    try:
        originalImage, data = turnToOriginal(medianifiedImage, n, bitStream)
        copyOfRecieved = np.array(originalImage, dtype = np.uint8)
    except:
        print("\n\n----------------------------------------------------------\nEMBEDDING CAPACITY EXCEEDED!! INCREASE VALUE OF N!!")
        return
    print("\n\n----------------------------------------------------------\nImage and data recived")
    a = True
    # originalImage -= 1
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if copyOfImage[i][j] != originalImage[i][j]:
                a = False
                break
    p = Psnr(copyOfMedian, copyOfEmbeddedImage)
    cv2.imshow("original-picture",copyOfImage)
    cv2.imshow("original-picture-median-Subtracted",copyOfMedian)
    cv2.imshow("embedded-picture-median-Subtracted",copyOfEmbeddedImage)
    cv2.imshow("finally-extracted-image-at-reciever-end",copyOfRecieved)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("\n\n----------------------------------------------------------\nAre original and recived images same: ", a)
    print("The data recived is", data)
    print("psnr value is: ",p)

starterFunction()