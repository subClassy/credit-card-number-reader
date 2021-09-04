import os
import random

import cv2
import numpy as np

cc1 = cv2.imread('credit_card_font/creditcard_digits1.jpg', 0)
cc2 = cv2.imread('credit_card_font/creditcard_digits2.jpg', 0)

_, th1 = cv2.threshold(cc1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, th2 = cv2.threshold(cc2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
 
def makedir(directory):
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

def pre_process(image, inv = False):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if inv == False:
        _, threshold = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, threshold = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    return cv2.resize(threshold, (32,32), interpolation = cv2.INTER_AREA)

# Hard coded for the given image
region = [(2, 19), (50, 72)]

top_left_y = region[0][1]
bottom_right_y = region[1][1]
top_left_x = region[0][0]
bottom_right_x = region[1][0]

for i in range(0,10):   
    if i > 0:
        # Numbers have been found through trial and error
        top_left_x = top_left_x + 59
        bottom_right_x = bottom_right_x + 59

    roi = th1[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    print("Augmenting Digit - ", str(i))
    for j in range(0,2000):
        roi2 = DigitAugmentation(roi)
        roi_otsu = pre_process(roi2, inv = True)
        cv2.imwrite("./credit_card/train/"+str(i)+"./img_1_"+str(j)+".jpg", roi_otsu)
        
        # For test images
        if (j < 200):
            roi2 = DigitAugmentation(roi)
            roi_otsu = pre_process(roi2, inv = True)
            cv2.imwrite("./credit_card/test/"+str(i)+"./img_1_"+str(j)+".jpg", roi_otsu)

# Hard coded numbers for second image
region = [(0, 0), (35, 48)]

top_left_y = region[0][1]
bottom_right_y = region[1][1]
top_left_x = region[0][0]
bottom_right_x = region[1][0]

for i in range(0,10):   
    if i > 0:
        top_left_x = top_left_x + 35
        bottom_right_x = bottom_right_x + 35

    roi = th2[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    print("Augmenting Digit - ", str(i))
    for j in range(0,2000):
        roi2 = DigitAugmentation(roi)
        roi_otsu = pre_process(roi2, inv = False)
        cv2.imwrite("./credit_card/train/"+str(i)+"./img_2_"+str(j)+".jpg", roi_otsu)
        
        # For test images
        if (j < 200):
            roi2 = DigitAugmentation(roi)
            roi_otsu = pre_process(roi2, inv = True)
            cv2.imwrite("./credit_card/test/"+str(i)+"./img_2_"+str(j)+".jpg", roi_otsu)


