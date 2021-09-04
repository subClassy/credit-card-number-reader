import cv2
import os
import numpy as np
from random import random

cc1 = cv2.imread('credit_card_font/creditcard_digits1.jpg', 0)
cc2 = cv2.imread('credit_card_font/creditcard_digits2.jpg', 0)

_, th2 = cv2.threshold(cc2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
 
def makedir(directory):
    """Creates a new directory if it does not exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        return None, 0
    
for i in range(0, 10):
    directory_name = "./credit_card/train/" + str(i)
    print(directory_name)
    makedir(directory_name) 
 
for i in range(0, 10):
    directory_name = "./credit_card/test/" + str(i)
    print(directory_name)
    makedir(directory_name)

def DigitAugmentation(frame, dim = 32):
    frame = cv2.resize(frame, None, fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    random_num = np.random.randint(0,9)
 
    if (random_num % 2 == 0):
        frame = add_noise(frame)
    if(random_num % 3 == 0):
        frame = pixelate(frame)
    if(random_num % 2 == 0):
        frame = stretch(frame)
    frame = cv2.resize(frame, (dim, dim), interpolation = cv2.INTER_AREA)
 
    return frame 

def add_noise(image):
    prob = random.uniform(0.01, 0.05)
    rnd = np.random.rand(image.shape[0], image.shape[1])
    noisy = image.copy()
    noisy[rnd < prob] = 0
    noisy[rnd > 1 - prob] = 1
    
    return noisy

def pixelate(image):
    dim = np.random.randint(8, 12)
    image = cv2.resize(image, (dim, dim), interpolation = cv2.INTER_AREA)
    image = cv2.resize(image, (16, 16), interpolation = cv2.INTER_AREA)
    
    return image

def stretch(image):
    ran = np.random.randint(0, 3) * 2
    
    if np.random.randint(0, 2) == 0:
        frame = cv2.resize(image, (32, ran + 32), interpolation = cv2.INTER_AREA)
        return frame[int(ran/2):int(ran+32) - int(ran/2), 0:32]
    else:
        frame = cv2.resize(image, (ran+32, 32), interpolation = cv2.INTER_AREA)
        return frame[0:32, int(ran/2):int(ran+32) - int(ran/2)]

