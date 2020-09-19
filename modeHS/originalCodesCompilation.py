# For reference functions

import numpy as np
import cv2
import matplotlib.pyplot as plt

def encrypt(image, key):
    image = np.asarray(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i][j] |= key
    return image

def Nalpha(a,b):
    if a<=b:
        return (2 ** a - 1)
    else:
        return (2 ** b - 1) * (2 ** (b - a))

def decodeClassify(matrix, a, b, first = False):
    # Assuming sub-matrices of size 2 x 2
    # taking 0,0 to be the reference pixel
    refVal = matrix[0][0]
    # 0: reference pixel
    # 1: special pixel
    # 2: embedding pixel
    # 3: non-embedding pixel
    category = [[0,0],[0,0]]
    e1 = matrix[0][1]
    e2 = matrix[1][0]
    e3 = matrix[1][1]
    if e1//(2**(8-b)) == 0:
        category[0][1] = 3
    else:
        category[0][1] = 2
    if e2//(2**(8-b)) == 0:
        category[1][0] = 3
    else:
        category[1][0] = 2
    if e3//(2**(8-b)) == 0:
        category[1][1] = 3
    else:
        category[1][1] = 2
    if first :
        category[0][1] = 1
    return category
    
def classify(matrix, a, b, first = False):
    # Assuming sub-matrices of size 2 x 2
    # taking 0,0 to be the reference pixel
    refVal = matrix[0][0]
    # 0: reference pixel
    # 1: special pixel
    # 2: embedding pixel
    # 3: non-embedding pixel
    category = [[0,0],[0,0]]
    e1 = matrix[0][1]
    e2 = matrix[1][0]
    e3 = matrix[1][1]
    alpha = Nalpha(a,b)
    left = int(-alpha/2)
    right = int((alpha-1)/2)
    if e1<left or e1>right:
        category[0][1] = 3
    else:
        category[0][1] = 2
    if e2<left or e2>right:
        category[1][0] = 3
    else:
        category[1][0] = 2
    if e3<left or e3>right:
        category[1][1] = 3
    else:
        category[1][1] = 2
    if first:
        category[0][1] = 1
    return category

def PBTLDE(coverImage, secretDataStream, alpha, beta):
    coverImage = np.asarray(coverImage)
    embeddedImage = np.array(coverImage)
    p = [[],[],[],[]]
    subcategories = nalpha(alpha, beta)
    eStep = 512/subcategories
    payload = "{0:b}".format(coverImage[0][1])
    for i in range(256):
        for j in range(256):
            matrix = np.array(coverImage[2*i:2*i+2,2*j:2*j+2])
            ematrix = np.array(matrix)
            refPixel = matrix[0][0]
            for a in range(2):
                for b in range(2):
                    if a==0 and b==0:
                        ematrix[a][b] -= refPixel
            categories = classify(ematrix,((i==0) and (j==0)))
            for a in range(2):
                for b in range(2):
                    p[categories[a][b]].append([i,j])
                    if categories[a][b]==3:
                        payload += ("{0:b}".format(matrix[a][b]))[:beta]
                        embeddedImage[2*i+a][2*j+b] %= 2**(8-beta)
    payload += secretDataStream
    for i in range(256):
        for j in range(256):
            matrix = np.array(coverImage[2*i:2*i+2,2*j:2*j+2])
            ematrix = np.array(matrix)
            refPixel = matrix[0][0]
            for a in range(2):
                for b in range(2):
                    if a==0 and b==0:
                        ematrix[a][b] -= refPixel
            categories = classify(ematrix,((i==0) and (j==0)))
            for a in range(2):
                for b in range(2):
                    if categories[a][b] == 2:
                        na = int((ematrix[a][b]+256)/eStep)
                        if len(payload)>alpha:
                            embeddedImage[2*i:a][2*j+b] = na * 2**(8-alpha) + int(payload[:alpha],2)
                            payload = payload[alpha:]
                        else:
                            embeddedImage[2*i:a][2*j+b] = na * 2**(8-alpha) + int(payload,2)
                            return embeddedImage
    return embeddedImage

def recover(embeddedImage):
    embeddedImage = np.array(embeddedImage)
    recoveredImage = np.array(embeddedImage)
    p = [[],[],[],[]]
    ab = embeddedImage[0][1]
    alpha = ab//16
    beta = ab%16
    subcategories = nalpha(alpha, beta)
    eStep = 512/subcategories
    payload = ""
    for i in range(256):
        for j in range(256):
            matrix = np.array(coverImage[2*i:2*i+2,2*j:2*j+2])
            refPixel = matrix[0][0]
            categories = decodeClassify(matrix,((i==0) and (j==0)))
            for a in range(2):
                for b in range(2):
                    p[categories[a][b]].append([i,j])
                    if categories[a][b]==2:
                        # embedding pixels
                        payload += ("{0:b}".format(matrix[a][b]))[alpha:]
                        na = matrix[a][b]//(2**(8-alpha))
                        recoveredImage[2*i+a][2*j+b] = na * eStep - 256
    recoveredImage[0][1] = int(payload[:8],2)
    payload = payload[8:]
    for i in range(256):
        for j in range(256):
            matrix = np.array(coverImage[2*i:2*i+2,2*j:2*j+2])
            refPixel = matrix[0][0]
            categories = decodeClassify(matrix,((i==0) and (j==0)))
            for a in range(2):
                for b in range(2):
                    if categories[a][b] == 2:
                        recoveredImage[2*i + a][2*j + b] += refPixel
                    elif categories[a][b] == 3:
                        recoveredImage[2*i+a][2*j+b] += int(payload[:beta],2)*2**(8-beta)
                        payload = payload[beta:]
    secretDataStream = payload
    print("recovered secret Bit Stream is: ", secretDataStream)
    return recoveredImage


image = cv2.resize(cv2.imread("coverImage.jpeg"),(512,512))
key = 73
image = encrypt(image,key)
message = "010101010101010010110101010101010010101011010100100101010110011100011000"
alpha = int(input("Enter the value of alpha"))
beta = int(input("Enter the value of beta"))
embedded = PBTLDE(image, message, alpha, beta)

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

import sys
import cv2
from PIL import Image
import numpy as np
import math
import os

def Psnr(img1, img2):
    img1=cv2.imread(img1)
    img2=cv2.imread(img2)
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

 def in_img(self,address=''):
        if(address==''):
            im = Image.open(self.loc1)
            self.image1.setPixmap(QPixmap(self.loc1).scaled(640, 360, Qt.IgnoreAspectRatio, Qt.FastTransformation))
            pix = im.load()
            height,width=im.size # Get the width and hight of the image for iterating over
            temp=np.zeros((width,height))
            #calculating frequency of grayscale value in the image and ploting bar graph
            frequency=list()
            col=list()
            x=list()
            for i in range(256):
                frequency.append(0)
                x.append(i)
                col.append('lime')
            for i in range(height):
                for j in range(width):
                    frequency[pix[i,j]]+=1
            
            #finding peak point and minimum point
            maxp=0
            for i in range(256):
                if(frequency[i]>frequency[maxp]):
                    maxp=i
            if(maxp==0):
                maxp2=1
            else:
                maxp2=0
            for i in range(256):
                if(frequency[i]>frequency[maxp2] and frequency[i]<frequency[maxp]):
                    maxp2=i
            print(maxp,maxp2)
            if(maxp>maxp2): 
                for i in range(maxp2,maxp+1):
                    col[i]='cyan'
            else:
                for i in range(maxp,maxp2+1):
                    col[i]='cyan'
            self.g1.plot(x=x,y=frequency,title='initial img',color=col)
            #finding and storing coordinates of minimum value
            self.cor=list()
            for i in range(height):
                for j in range(width):
                    if(pix[i,j]==maxp2):
                       self.cor.append([i,j])
            
            
            #histogram shifting
            self.flag=0
            if(maxp-maxp2==1):
                self.flag=1
                maxp2-=1
            elif(maxp-maxp2==-1):
                maxp2+=1
                self.flag=-1
            
            if(maxp<maxp2):
                for i in range(height):
                    for j in range(width):
                        if(pix[i,j]<maxp2 and pix[i,j]>maxp):
                            temp[j][i]=255
                            pix[i,j]+=1
            else:
                for i in range(height):
                    for j in range(width):
                        if(pix[i,j]>maxp2 and pix[i,j]<maxp):
                            temp[j][i]=255
                            pix[i,j]-=1
            maxp2+=self.flag
        
                        
            #recalculating frequency
            for i in range(256):
                frequency[i]=0
            for i in range(height):
                for j in range(width):
                    frequency[pix[i,j]]+=1
            
            
            self.maxp=maxp
            self.maxp2=maxp2
            
            #saving the image
            cv2.imwrite('./temp/shift.png',temp)
            self.g2.plot(x=x,y=frequency,color='lime',title=('after histogram shifting\n'+'psnr='+str(round(Psnr(self.loc1,'./temp/shift.png'),4))))
            self.image2.setPixmap(QPixmap('./temp/shift.png').scaled(640, 360, Qt.IgnoreAspectRatio, Qt.FastTransformation))
        else:
            im = Image.open(self.loc1)
            pix = im.load()
            height,width=im.size # Get the width and hight of the image for iterating over
            
            maxp=self.maxp
            maxp2=self.maxp2
            bstring=''
            for i in range(self.size):
                for j in range(self.size):
                    if(self.in_wm[i][j]):
                        bstring+='1'
                    else:
                        bstring+='0'
         
            
            #histogram shifting
            maxp2-=self.flag
            if(maxp<maxp2):
                for i in range(height):
                    for j in range(width):
                        if(pix[i,j]<maxp2 and pix[i,j]>maxp):
                            pix[i,j]+=1
            else:
                for i in range(height):
                    for j in range(width):
                        if(pix[i,j]>maxp2 and pix[i,j]<maxp):
                            pix[i,j]-=1
            maxp2+=self.flag
                
            #writing binary data into image
            k=0
            for i in range(height):
                for j in range(width):
                    if(pix[i,j]==maxp and k<len(bstring)):
                        if(bstring[k]=='1' and maxp2>maxp):
                            pix[i,j]+=1
                        elif(bstring[k]=='1' and maxp2<maxp):
                            pix[i,j]-=1
                        k+=1
                        
            #recalculating frequency
            frequency=list()
            x=list()
            for i in range(256):
                frequency.append(0)
                x.append(i)
            for i in range(height):
                for j in range(width):
                    frequency[pix[i,j]]+=1
            
            
            self.maxp=maxp
            self.maxp2=maxp2            
            
            
            #saving the image
            im.save('./temp/embeded.png')
            
            psnr=cv2.PSNR(cv2.imread(self.loc1),cv2.imread('./temp/embeded.png'))
            psnr=round(psnr,4)
            self.g3.plot(x=x,y=frequency,color='lime',title=('after writing data    '+'psnr='+str(round(Psnr(self.loc1,'./temp/embeded.png'),4))))
        