from pathlib import Path
import glob
# from numba import jit, cuda 
import numpy as np
import cv2
import sys
from scipy import sqrt, pi, arctan2, cos, sin
from scipy.ndimage import uniform_filter
import math

# quantize rgb from 256*256*256 to 4*4*4 color bins
def convert64(pixel):
    return (convert4(pixel[0]),convert4(pixel[1]),convert4(pixel[2]))

# convert one color in the range 0-255 to range 0-3
def convert4(val):
    if(val<64):
        return 0
    if(val<128):
        return 1
    if(val<192):
        return 2
    if(val<256):
        return 3

# return an index for each of 64 color bins
def convertColorToIndex(color):
    return 16*color[0] + 4*color[1] + 1*color[2] 

# autocorrelogram parent function
def autocorrelogram(img, colorDict):
    colorbins = []
    for i in range(0,4):
        for j in range(0,4):
            for k in range(0,4):
                color = (i,j,k)
                colorbins.append(color)
    ac = np.empty([64, 5])
    distances = [1,3,5,7,9]
    for color in colorbins:
        for d in range(0,5):
            distance = distances[d]
            acgs= autocorrelogramStep(distance, color, img, colorDict)
            ac[convertColorToIndex(color)][d]  = acgs
    return ac

# helper function that calculates value for a given color c and distance k
def autocorrelogramStep(k,c,img, colorDict):
    if not convertColorToIndex(c) in colorDict:
        return 0
    else:
        return (Tfunction(c,k,img))/(colorDict[convertColorToIndex(c)]*8*k)

# @jit
def Tfunction(c,k,img):
    ans = 0
    verticalDict = {}
    horizontalDict = {}
    step = 15 # step set to 15 to allow for quick computation. Ideally step should be k/2
    # step = math.ceil(k/2)
    for x in range(0,len(img), step):
        for y in range(0,len(img[0]), step):
            j = img[x][y]
            if(j[0] == c[0] and j[1] == c[1] and j[2] == c[2]):
                a1 = lambdaHorizontal(x-k,y+k,c, 2*k , img, horizontalDict)
                b1 = lambdaHorizontal(x-k,y-k,c, 2*k , img, horizontalDict)
                c1 = lambdaVerticle(x-k,y-k+1,c, (2*k)-2 , img, verticalDict)
                d1 = lambdaVerticle(x+k,y-k+1,c, (2*k)-2 , img, verticalDict)
                ans+= (a1+b1+c1+d1)
    return ans

# @jit
def lambdaHorizontal(x,y,c,k, img, horizontalDict):
    if k!=0:
        if not (x,y, convertColorToIndex(c), k-1) in horizontalDict:
            horizontalDict[(x,y,convertColorToIndex(c), k-1)] = lambdaHorizontal(x,y,c,k-1,img, horizontalDict)
        if not (x+k, y,c, 0) in horizontalDict:
            horizontalDict[(x+k, y,convertColorToIndex(c), 0)] = lambdaHorizontal(x+k,y,c,0,img, horizontalDict)
        return  horizontalDict[(x,y, convertColorToIndex(c),k-1)] + horizontalDict[(x+k, y, convertColorToIndex(c),0)] 
    else:
        if(not isPixel(x,y, img)):
            return 0
        if (img[x][y][0] == c[0] and img[x][y][1] == c[1] and img[x][y][2] == c[2]):
            return 1
        else:
            return 0

# @jit
def lambdaVerticle(x,y,c,k, img, verticalDict):
    if k!=0:
        if not (x,y,c,k-1) in verticalDict:
            verticalDict[(x,y,convertColorToIndex(c),k-1)] = lambdaVerticle(x,y,c,k-1,img, verticalDict)
        if not (x, y+k, 0) in verticalDict:
            verticalDict[(x, y+k, convertColorToIndex(c),0)] = lambdaVerticle(x, y+k,c,0,img, verticalDict)
        return  verticalDict[(x,y,convertColorToIndex(c),k-1)] + verticalDict[(x, y+k,convertColorToIndex(c), 0)]
    else:
        if(not isPixel(x,y, img)):
            return 0
        if (img[x][y][0] == c[0] and img[x][y][1] == c[1] and img[x][y][2] == c[2]):
            return 1
        else:
            return 0

# Helper function to make sure pixel queried actually is within image range
def isPixel(x,y,img):
    if(x<0 or y<0):
        return False
    elif(x>=len(img)):
        return False
    elif(y >= len(img[0])):
        return False
    else:
        return True

# import all images from images folder
images = (glob.glob("HW1/images/*.jpg"))
from PIL import Image
import pickle

autocorrelogramRetrieval = {}
# run color autocorrelogram on each image
for ims in range(0,len(images)):
    image = images[ims]
    # print(str(ims)+ " of " + str(len(images))+ ": "+ str(image))
    #resize image
    cimg = np.array(Image.open(image).resize((300,300)))
    colorDict = {}
    for i in cimg:
        for j in i:
            j[0], j[1], j[2]  = convert64(j)
            colorind = convertColorToIndex(j)
            if not colorind in colorDict:
                colorDict[colorind] = 1
            else:
                colorDict[colorind] = colorDict[colorind] + 1

    imageCg = (autocorrelogram(cimg, colorDict))
    autocorrelogramRetrieval[image] = imageCg
    # dump data into correlogram
    pickle.dump(autocorrelogramRetrieval, open("autocorrelogram.p", "wb"))
