import numpy as np
import cv2
from math import floor, ceil

def toBinary(a):
    a = "{0:b}".format(a)
    a = "0"*(8-len(a)) + a
    if len(a) != 8:
        print(a)
    return a

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
        return ((2 ** b) - 1) * (2 ** (a - b))

def classify(matrix, a, b, first = False):
    # Assuming sub-matrices of size 2 x 2
    # taking 0,0 to be the reference pixel
    refVal = matrix[0][0]
    # 0: reference pixel
    # 1: special pixel
    # 2: embedding pixel
    # 3: non-embedding pixel
    category = [[0,0],[0,0]]
    n_alpha = Nalpha(a,b)
    left = ceil(-n_alpha/2)
    right = floor((n_alpha-1)/2)
    for i in range(2):
        for j in range(2):
            if i+j != 0:
                if matrix[i][j]<left or matrix[i][j]>right:
                    category[i][j] = 3
                else:
                    # print("HERE!!")
                    category[i][j] = 2
    if first:
        # print("SPECIALLY HERE")
        category[0][1] = 1
    return category

def PBTLDE(coverImage, secretDataStream, alpha, beta):
    coverImage = np.asarray(coverImage, dtype = np.int16)
    embeddedImage = np.array(coverImage, dtype = np.int16)
    # p will store the indexes (i,j) pairs or locations of different categories of pixels i.e. 
    # p[0]: reference pixels
    # p[1]: special pixel
    # p[2]: embedding pixels
    # p[3]: non-embedding pixels
    p = [[],[],[],[]]
    # payload will be constructed of datastream + breaker + special pixel bits + total beta bits embedded
    payload = secretDataStream
    # breaker
    payload += "0011111111111100"
    payload += toBinary(coverImage[0][1])
    # for checking if ceil(-nalpha/2) to floor((nalpha-1)/2) takes more than alpha bits
    # alph = Nalpha(alpha,beta)
    # left = ceil(-alpha/2)
    # right = floor((alpha-1)/2)
    # print("Nalpha: {}, left: {}, right: {}".format(alph, left, right))
    # This is however not required as the range of ceil(-Nalpha/2) to floor((Nalpha-1)/2) was checked to be <= Nalpha for alpha in range(8) and beta in range(8)
    for i in range(256):
        for j in range(256):
            # matrix is a copy of the elements in a 2x2 subdivision of the image
            matrix = np.array(coverImage[2*i:2*i+2,2*j:2*j+2])
            # ematrix will store the values of pixel-referencePixel for categorising into Pn and Pe
            ematrix = np.array(matrix)
            refPixel = matrix[0][0]
            for a in range(2):
                for b in range(2):
                    if (a + b) != 0:
                        ematrix[a][b] -= refPixel
                        embeddedImage[2*i+a][2*j+b] = ematrix[a][b] 
            # categorising into type of pixel
            categories = classify(ematrix, alpha, beta, (i+j == 0))
            for a in range(2):
                for b in range(2):
                    # adding location per category to corresponding category array
                    p[categories[a][b]].append([2*i+a,2*j+b])
                    if categories[a][b]==3:
                        # adding first beta bits of Pn type pixels to the payload
                        payload += (toBinary(matrix[a][b]))[:beta]

    # final breaker of payload
    payload += "0011111111111100"
    totalBitsToEmbed = len(payload)
    totalCapacity = len(p[2])*(8-alpha)
    capacityPerPixel = totalCapacity/(512*512)
    print("Bits to be embedded: {}\nEmbedding capacity:\n\tTotal no. of bits: {}\n\tEmbedding capacity per pixel: {}".format(totalBitsToEmbed, totalCapacity, capacityPerPixel))
    # Implementing the parametric binary tree labeling system for the 2 cases : alpha<=beta and alpha>beta
    # Also embedding data into 8-alpha bits of the Pe pixels
    if alpha<=beta:
        toEncode = ceil(-1*Nalpha(alpha,beta)/2)-1
        for [i,j] in p[2]:
            embeddedImage[i][j] -= toEncode
            embeddedImage[i][j] *= 2**(8-alpha)
            embeddedImage[i][j] += int(payload[:8-alpha],2)
            payload = payload[8-alpha:]
            if len(payload) < (8-alpha):
                payload += "1" * (8-alpha-len(payload))
    else:
        toEncode = ceil(-1*Nalpha(alpha,beta)/2)
        for [i,j] in p[2]:
            embeddedImage[i][j] -= toEncode
            embeddedImage[i][j] *= 2**(8-alpha)
            embeddedImage[i][j] += int(payload[:8-alpha],2)
            embeddedImage[i][j] += 2**(8-beta)
            payload = payload[8-alpha:]
            if len(payload) < (8-alpha):
                payload += "1" * (8-alpha-len(payload))

    # Setting first beta bits of Pn pixels to 000..
    for [i,j] in p[3]:
        embeddedImage[i][j] = coverImage[i][j]%(2**(8-beta))
    # saving the value of alpha and beta in the special pixel
    [i,j] = p[1][0]
    embeddedImage[i][j] = alpha * 16 + beta

    # Type conversion back to 8 bit numbers
    embeddedImage = np.asarray(embeddedImage, dtype = np.uint8)
    return embeddedImage

def recover(embeddedImage):
    embeddedImage = np.asarray(embeddedImage, dtype = np.int16)
    recoveredImage = np.array(embeddedImage, dtype = np.int16)
    # p is the same as before and alpha and beta are derived from the predefined special pixel position
    p = [[],[],[],[]]
    ab = embeddedImage[0][1]
    alpha = ab//16
    beta = ab%16
    # Labeling pixels based on starting beta bits
    payload = ""
    for i in range(256):
        for j in range(256):
            matrix = np.array(recoveredImage[2*i:2*i+2,2*j:2*j+2])
            refPixel = matrix[0][0]
            for a in range(2):
                for b in range(2):
                    if (a+b) != 0:
                        if (2*i+a == 0) and (2*j + b == 1):
                            p[1].append([2*i+a,2*j+b])
                        elif (toBinary(matrix[a][b])[:beta] != ("0"*beta)):
                            p[2].append([2*i+a,2*j+b])
                        else:
                            p[3].append([2*i+a,2*j+b])
                    else:
                        p[0].append([2*i+a,2*j+b])
    # Extracting payload
    for [i,j] in p[2]:
        payload += toBinary(embeddedImage[i][j])[alpha:]

    # extracting data stream from recovered payload
    dataEnd = payload.index("0011111111111100")
    data = payload[:dataEnd]
    payload = payload[dataEnd+16:]
    specialElement = int(payload[:8],2)
    payload = payload[8:]
    betaBitsEnd = payload.index("0011111111111100")
    
    # Replacing first beta bits of the Pn pixels with beta bits from the payload
    for [i,j] in p[3]:
        recoveredImage[i][j] += int(payload[:beta],2)*(2**(8-beta))
        payload = payload[beta:]

    # Recovering Pe pixel values
    if alpha<=beta:
        toEncode = ceil(-1*Nalpha(alpha,beta)/2)-1
        for [i,j] in p[2]:
            recoveredImage[i][j] = recoveredImage[i][j]//(2**(8-alpha))
            recoveredImage[i][j] += toEncode
    else:
        toEncode = ceil(-1*Nalpha(alpha,beta)/2)
        for [i,j] in p[2]:
            recoveredImage[i][j] -= 2**(8-beta)
            recoveredImage[i][j] = recoveredImage[i][j]//(2**(8-alpha))
            recoveredImage[i][j] += toEncode
    for [i,j] in p[2]:
        refPixel = recoveredImage[i-i%2][j-j%2]
        recoveredImage[i][j] += refPixel

    # Finally setting special pixel value back to original
    recoveredImage[0][1] = specialElement

    # Converting value size back to 8 bits
    recoveredImage = np.asarray(recoveredImage, dtype = np.uint8)
    print("recovered secret Bit Stream is: ", data)
    return recoveredImage, data


image = cv2.resize(cv2.imread("test.tiff",0),(512,512))
img = np.array(image, dtype = np.uint8)
# key = 73
# image = encrypt(image,key)

message = input("Enter message bit sequence like: 010101010101010010110100...   ")
print("Valid values of alpha and beta for the image of the man are:")
valids = [(2,1),(3,1),(3,2),(3,3),(4,1),(4,2),(4,3),(4,4),(4,5),(4,6),(4,7),(5,1),(5,2),(5,3),(5,4),(5,5),(5,6),(5,7),(6,1),(6,2),(6,3),(6,4),(6,5),(6,6),(6,7),(7,1),(7,2),(7,3),(7,4),(7,5),(7,6),(7,7)]
for valid in valids:
    print(valid)
alpha = int(input("Enter the value of alpha:     "))
beta = int(input("Enter the value of beta:    "))

# For cross - checking the valid values of alpha and beta:
# for alpha in range(8):
#     for beta in range(8):
#         try:
#             embedded = PBTLDE(image, message, alpha, beta)
#             recovered, data = recover(embedded)
#         except:
#             print("negative embedding degree for alpha = {} and beta = {}, try again with different values of alpha and beta".format(alpha,beta))
embedded = PBTLDE(image,message,alpha,beta)
embCopy = np.array(embedded)
recovered, data = recover(embedded)
count = 0
for i in range(512):
    for j in range(512):
        if (img[i][j] != recovered[i][j]):
            count+=1
print("END of recieving, the number of mismatches= ", count)
cv2.imshow("Original Image",img)
cv2.imshow("Embedded Image",embCopy)
cv2.imshow("Recieved Image",recovered)
cv2.waitKey(0)
cv2.destroyAllWindows()