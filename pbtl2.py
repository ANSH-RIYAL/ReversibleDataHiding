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
    print("Pe: 1,0: ", embeddedImage[511][510])
    print("Pn: 1,4: ", embeddedImage[1][4], toBinary(embeddedImage[1][4]))
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
            # print(ematrix)
            categories = classify(ematrix, alpha, beta, (i+j == 0))
            for a in range(2):
                for b in range(2):
                    # adding location per category to corresponding category array
                    p[categories[a][b]].append([2*i+a,2*j+b])
                    if categories[a][b]==3:
                        # adding first beta bits of Pn type pixels to the payload
                        payload += (toBinary(matrix[a][b]))[:beta]

    print("Pe: 1,0: ", embeddedImage[511][510])
    print("Pn: 1,4: ", embeddedImage[1][4])

    if [511,510] in p[3]:
        print("Exists in 3!\n\n\n\n")
    if [511,510] in p[2]:
        print("Exists in 2!\n\n\n\n")
    if [511,510] in p[1]:
        print("Exists in 1!\n\n\n\n")
    if [511,510] in p[0]:
        print("Exists in 0!\n\n\n\n")

    # final breaker of payload
    payload += "0011111111111100"
    # print(payload[:200])
    # print(len(p[2]))
    # shifting the value of embeddedImage pixels to +ve side so that they take alpha bits only
    if alpha<=beta:
        toEncode = ceil(-1*Nalpha(alpha,beta)/2)-1
        for [i,j] in p[2]:
            if i==511 and j == 510:
                print(Nalpha(alpha,beta),-toEncode,embeddedImage[i][j])
            embeddedImage[i][j] -= toEncode
            # if (embeddedImage[i][j] < 1) or (embeddedImage[i][j]>2**alpha):
            #     print("Gadbad is here!")
            if i==511 and j == 510:
                print(embeddedImage[i][j])
            embeddedImage[i][j] *= 2**(8-alpha)
            if i==511 and j == 510:
                print(toBinary(embeddedImage[i][j]))
            embeddedImage[i][j] += int(payload[:8-alpha],2)
            if i==511 and j == 510:
                print(toBinary(embeddedImage[i][j]))
            payload = payload[8-alpha:]
            if len(payload) < (8-alpha):
                payload += "1" * (8-alpha-len(payload))
    else:
        toEncode = ceil(-1*Nalpha(alpha,beta)/2)
        for [i,j] in p[2]:
            if i==511 and j == 510:
                print(Nalpha(alpha,beta),-toEncode,embeddedImage[i][j])
            embeddedImage[i][j] -= toEncode
            # if (embeddedImage[i][j] < 0) or (embeddedImage[i][j]>2**(alpha-beta)):
            #     print("Gadbad is here!")
            if i==511 and j == 510:
                print(embeddedImage[i][j])
            embeddedImage[i][j] *= 2**(8-alpha)
            if i==511 and j == 510:
                print(toBinary(embeddedImage[i][j]))
            embeddedImage[i][j] += int(payload[:8-alpha],2)
            if i==511 and j == 510:
                print(toBinary(embeddedImage[i][j]))
            embeddedImage[i][j] += 2**(8-beta)
            if i==511 and j == 510:
                print(toBinary(embeddedImage[i][j]))
                print(embeddedImage[i][j])
            payload = payload[8-alpha:]
            if len(payload) < (8-alpha):
                payload += "1" * (8-alpha-len(payload))

    print("Pe: 1,0: ", toBinary(embeddedImage[511][510]))
    print("Pn: 1,4: ", toBinary(embeddedImage[1][4]))

    # for [i,j] in p[2]:
    #     embeddedImage[i][j] -= toEncode
    #     if (embeddedImage[i][j] < 1):
    #         print("Gadbad is here!")
    #     embeddedImage[i][j] *= 2**(8-alpha)
    #     embeddedImage[i][j] += int(payload[:8-alpha],2)
    #     payload = payload[8-alpha:]
    #     if len(payload) < (8-alpha):
    #         payload += "1" * (8-alpha-len(payload))
    for [i,j] in p[3]:
        embeddedImage[i][j] = coverImage[i][j]%(2**(8-beta))
    [i,j] = p[1][0]
    embeddedImage[i][j] = alpha * 16 + beta
    embeddedImage = np.asarray(embeddedImage, dtype = np.uint8)

    print("Pe: 1,0: ", embeddedImage[511][510])
    print("Pn: 1,4: ", embeddedImage[1][4])

    # print("payload left: ", payload)
    # print("Embedding pixels")
    # for [i,j] in p[2][:5]:
    #     print(i,j)
    #     print(toBinary(embeddedImage[i][j]))
    # print("Non embedding pixels")
    # for [i,j] in p[3][:5]:
    #     print(i,j)
    #     print(toBinary(embeddedImage[i][j]))
    # print(toBinary(embeddedImage[1][1]))
    return embeddedImage

def recover(embeddedImage):
    embeddedImage = np.asarray(embeddedImage, dtype = np.int16)
    recoveredImage = np.array(embeddedImage, dtype = np.int16)
    # print("Zero here!")
    print("Pe: 1,0: ", embeddedImage[511][510])
    print("Pn: 1,4: ", embeddedImage[1][4])

    p = [[],[],[],[]]
    ab = embeddedImage[0][1]
    alpha = ab//16
    beta = ab%16
    # print(alpha,beta,"\n\n\n")
    payload = ""
    # print("1,1: ",toBinary(embeddedImage[1][1]))
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

    print("Pe: 1,0: ", recoveredImage[511][510])
    print("Pn: 1,4: ", recoveredImage[1][4])

    for [i,j] in p[2]:
        # print("here")
        payload += toBinary(embeddedImage[i][j])[alpha:]
    if [511,510] in p[3]:
        print("Exists in 3!\n\n\n\n")
    if [511,510] in p[2]:
        print("Exists in 2!\n\n\n\n")
    if [511,510] in p[1]:
        print("Exists in 1!\n\n\n\n")
    if [511,510] in p[0]:
        print("Exists in 0!\n\n\n\n")
    # print(p[2][:5])
    # for [i,j] in p[2][:5]:
    #     # print(toBinary())
    #     print(i,j)
    #     print(toBinary(embeddedImage[i][j]))
    
    # print("Payload recovered:\n",payload[:200])
    
    dataEnd = payload.index("0011111111111100")
    data = payload[:dataEnd]
    payload = payload[dataEnd+16:]
    specialElement = int(payload[:8],2)
    payload = payload[8:]
    betaBitsEnd = payload.index("0011111111111100")
    
    for [i,j] in p[3]:
        recoveredImage[i][j] += int(payload[:beta],2)*(2**(8-beta))
        payload = payload[beta:]

    print("Pe: 1,0: ", recoveredImage[511][510])
    print("Pn: 1,4: ", recoveredImage[1][4])
    
    if alpha<=beta:
        toEncode = ceil(-1*Nalpha(alpha,beta)/2)-1
        for [i,j] in p[2]:
            recoveredImage[i][j] = recoveredImage[i][j]//(2**(8-alpha))
            recoveredImage[i][j] += toEncode
    else:
        toEncode = ceil(-1*Nalpha(alpha,beta)/2)
        for [i,j] in p[2]:
            if i==511 and j == 510:
                    print(Nalpha(alpha,beta),-toEncode,embeddedImage[i][j])
            recoveredImage[i][j] -= 2**(8-beta)
            if i==511 and j == 510:
                print(recoveredImage[i][j], toBinary(recoveredImage[i][j]))
            recoveredImage[i][j] = recoveredImage[i][j]//(2**(8-alpha))
            if i==511 and j == 510:
                print(recoveredImage[i][j], toBinary(recoveredImage[i][j]))
            recoveredImage[i][j] += toEncode
            if i==511 and j == 510:
                print(recoveredImage[i][j], toBinary(recoveredImage[i][j]))

    print("Pe: 1,0: ", recoveredImage[511][510])
    print("Pn: 1,4: ", recoveredImage[1][4])

    for [i,j] in p[2]:
        refPixel = recoveredImage[i-i%2][j-j%2]
        recoveredImage[i][j] += refPixel
    # for i in range(256):
    #     for j in range(256):
    #         refElement = recoveredImage[2*i][2*j]
    #         for a in range(2):
    #             for b in range(2):
    #                 if (a+b != 0):
    #                     recoveredImage[2*i+a][2*j+b] += refElement
    recoveredImage[0][1] = specialElement
    recoveredImage = np.asarray(recoveredImage, dtype = np.uint8)
    print("recovered secret Bit Stream is: ", data)

    print("Pe: 1,0: ", recoveredImage[511][510])
    print("Pn: 1,4: ", recoveredImage[1][4])

    return recoveredImage


image = cv2.resize(cv2.imread("test.tiff",0),(512,512))
img = np.array(image, dtype = np.uint8)
# key = 73
# image = encrypt(image,key)
# message = "010101010101010010110101010101010010101011010100100101010110011100011000"
message = "11001010101001010101010"
alpha = int(input("Enter the value of alpha"))
beta = int(input("Enter the value of beta"))
embedded = PBTLDE(image, message, alpha, beta)
recovered = recover(embedded)
count = 0
for i in range(512):
    for j in range(512):

        if (img[i][j] != recovered[i][j]): # and (i%2 == 0) and (j%2 == 0):
            print("FALSE",i,j, img[i][j], recovered[i][j])
            count+=1
print("END", count, 512*512*0.75)
print(recovered[511][510], img[511][510])

# for a in range(8):
#     for b in range(8):
#         n = Nalpha(a,b)
#         bracket = floor((n-1)/2) - ceil(-n/2)
#         print(bracket)
#         print(2**a,"\n\n")
#         if bracket>=2**a:
#             print("cannot be encoded in alpha bits")
