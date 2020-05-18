# reference from: https://projectsflix.com/opencv/laplacian-blob-detector-using-python/

import cv2
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import filters
from scipy import spatial
import math
import time

def LoG(sigma):
    #window size 
    n = np.ceil(sigma*6)
    y,x = np.ogrid[-n//2:n//2+1,-n//2:n//2+1]
    y_filter = np.exp(-(y*y/(2.*sigma*sigma)))
    x_filter = np.exp(-(x*x/(2.*sigma*sigma)))
    final_filter = (-(2*sigma**2) + (x*x + y*y) ) *  (x_filter*y_filter) * (1/(2*np.pi*sigma**4))
    return final_filter

def LoG_convolve(img, k, sigma):
    log_images = [] #to store responses
    for i in range(0,9):
        y = np.power(k,i) 
        sigma_1 = sigma*y #sigma 
        filter_log = LoG(sigma_1) #filter generation
        image = cv2.filter2D(img,-1,filter_log) # convolving image
        image = np.pad(image,((1,1),(1,1)),'constant') #padding 
        image = np.square(image) # squaring the response
        log_images.append(image)
    log_image_np = np.array([i for i in log_images]) # storing the #in numpy array
    return log_image_np
#print(log_image_np.shape)


def detect_blob(log_image_np,img,k,sigma):
    co_ordinates = [] 
    (height,width) = img.shape
    for i in range(1,height,4):
        for j in range(1,width,4):
            slice_img = log_image_np[:,i-1:i+2,j-1:j+2] 
            result = np.amax(slice_img)
            if result >= 0.07: 
                z,x,y = np.unravel_index(slice_img.argmax(),slice_img.shape)
                co_ordinates.append((i+x-1,j+y-1 )) #/,k**z*sigma /#
    return co_ordinates

def returncoordinates(imageLocation):
    img = cv2.imread(imageLocation,0) 
    k = 1.414
    sigma = 1.0
    img = img/255.0 
    img = cv2.resize(img, (500,500), interpolation = cv2.INTER_AREA)
    log_image_np = LoG_convolve(img,k,sigma)
    co_ordinates = list(set(detect_blob(log_image_np,img,k,sigma)))
    print(str(len(co_ordinates)) + " blobs in image " , end = " ")
    # fig, ax = plt.subplots()
    # nh,nw = img.shape
    # count = 0

    # ax.imshow(img, interpolation='nearest',cmap="gray")
    # for blob in co_ordinates:
    #     y,x,r = blob
    #     c = plt.Circle((x, y), r*1.414, color='red', linewidth=1.5, fill=False)
    #     ax.add_patch(c)
    # ax.plot()  
    # plt.show()
    return co_ordinates

import glob
images = (glob.glob("HW1/images/*.jpg"))
SIFTdict = {}
for i in range(len(images)):
    startTime = time.time()
    coordinates = (returncoordinates(images[i]))
    SIFTdict[images[i]] = coordinates
    print(str(i) +": " + str(time.time() - startTime) )
    # break
    
import pickle
pickle.dump(SIFTdict, open("SIFT.p", "wb"))

