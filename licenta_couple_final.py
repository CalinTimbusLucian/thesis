# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 20:00:33 2018

@author: Calin
"""

#CHAPTER 1
import cv2
import numpy as np
input = cv2.imread('input.jpg')
#cv2.imshow('Rabbits',input)
print(input)
print(input.shape)
#Height
print(input.shape[0])
#Width
print(input.shape[1])
#BGR
#B,G,R = input[0,0]
#print (B, G, R)

B, G, R = cv2.split(input)
print(B.shape)
#cv2.imshow("Red",R)
#cv2.imshow("Green",G)
#cv2.imshow("Blue",B)


#Remake the original image
#merged = cv2.merge([B,G,R])
#cv2.imshow("Merged",merged)

#merged = cv2.merge([B+100,G,R])
#cv2.imshow("Merged Blue",merged)
#cv2.waitKey()
#cv2.destroyAllWindows()
#Converting to HSV and displaying each channel separately
#hsv_image = cv2.cvtColor(input,cv2.COLOR_BGR2HSV)
#cv2.imshow('HSV Image',hsv_image)
#cv2.imshow('Hue channel',hsv_image[:,:,0])
#cv2.imshow('Saturation channel',hsv_image[:,:,1])
#cv2.imshow('Value Channel',hsv_image[:,:,2])
#cv2.waitKey()
#cv2.destroyAllWindows()

#Let's create a matrix of zeros
# with dimensions of the image H x W
"""
zeros = np.zeros(input.shape[:2],dtype="uint8")
cv2.imshow("Red",cv2.merge([zeros,zeros,R]))
cv2.imshow("Green",cv2.merge([zeros,G,zeros]))
cv2.imshow("Blue",cv2.merge([B,zeros,zeros]))
cv2.waitKey()
cv2.destroyAllWindows()
"""

"""
cv2.imwrite('output.jpg',input)
cv2.imwrite('output2.png',input)
#Or read with imread('input.jpg',0)
gray_image = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray Image',gray_image)
cv2.waitKey()
cv2.destroyAllWindows()
"""

import cv2
import numpy as np
#We need to import matplotlib to create our histogram plots
from matplotlib import pyplot as plt
image = cv2.imread('input.jpg')
historgram = cv2.calcHist([input],[0],None,[256],[0,256])
#We plot a historgram, ravel() flattens our image array(takes a two-dimensional input array and makes a one-dimensional array out of it)
plt.hist(input.ravel(),256,[0,256]);plt.show()

#Viewing separate color channels
color = ('r','g','b')
for i,col in enumerate(color):
        histogram2 = cv2.calcHist([input],[i],None,[256],[0,256])
        plt.plot(histogram2, color = col)
        plt.xlim([0,256])
plt.show()

#CHAPTER 2
#Operations on images
import cv2
import numpy as np
#Translation
image = cv2.imread('input.jpg')
#Store height and width of the image
height,width = image.shape[:2]
quarter_height,quarter_width = height/4, width/4
#T is our translation matrix
T = np.float32([[1,0,quarter_width],[0,1,quarter_height]])
#We use warpAffine to transform the image using the matrix, T
img_translation = cv2.warpAffine(image,T,(width,height))
cv2.imshow('Translation',img_translation)
cv2.waitKey()
cv2.destroyAllWindows()

#Rotations
import cv2 
import numpy as np
image = cv2.imread('input.jpg')
#Take the first two elements of the array
height, width = image.shape[:2]
#Divide by two to rotate the image around its centre
rotation_matrix = cv2.getRotationMatrix2D((width/2,height/2),90,.75)
rotated_image = cv2.warpAffine(image,rotation_matrix,(width,height))
cv2.imshow('Rotated Image',rotated_image)
cv2.waitKey()
cv2.destroyAllWindows()

#Re-sizing,Scaling and Interpolation
#Interpolation = method of constructing new data points within
#the range of a discrete set of known data points
import cv2 
import numpy as np
image = cv2.imread('input.jpg')
#Let's make the image 3/4 of its initial size
image_scaled = cv2.resize(image,None,fx=0.75,fy=0.75)
#Let's double the size of our image
image_doubled = cv2.resize(image,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
#Let's skew the resizing by setting the exact dimension
img_scaled_2 = cv2.resize(image,(900,400),interpolation=cv2.INTER_AREA)
cv2.imshow('Original image',image)
cv2.imshow('Scaling-Linear Interpolation',image_scaled)
cv2.imshow('Scaling-Cubic Interpolation',image_doubled)
cv2.imshow('Image scaled 2',img_scaled_2)
cv2.waitKey()
cv2.destroyAllWindows()

#Image Pyramids
#Pyramiding image refers to either upscaling(enlarging) and downsampling
#Scaling down reduces the height and width of the image by half
#This comes in useful when making object detectors that scales images each time it looks for objects
import cv2
image = cv2.imread('input.jpg')
smaller = cv2.pyrDown(image)
larger = cv2.pyrUp(image)
cv2.imshow('Smaller',smaller)
cv2.imshow('Larger',larger)
cv2.waitKey()
cv2.destroyAllWindows()

#Cropping images
import cv2
import numpy as np
image = cv2.imread('input.jpg')
height, width = image.shape[:2]
#Let's get the starting pixel coordinates (top left of cropping rectangle)
start_row,start_col = int(height * .25), int(width * .25)
#Let's get the ending pixel coordinates(bottom right)
end_row,end_col = int(height * .75), int(width* .75)
#Simply use the indexing to crop out the rectangle we desire
cropped = image[start_row:end_row,start_col:end_col]
cv2.imshow('Original image',image)
cv2.imshow('Cropped image',cropped)
cv2.waitKey()
cv2.destroyAllWindows()

#Arithmetic operations
import cv2
import numpy as np
image = cv2.imread('input.jpg')
#Create a matrix of ones, then multiply it by a scaler of 100
#This gives a matrix with same dimensions of our image with all values being 100
M = np.ones(image.shape,dtype="uint8") * 75

#We use this to add this matrix M to our image
#Notice the increase in brightness
added = cv2.add(image,M)
cv2.imshow("Added",added)

#Likewise we can also subtract
#Notice the decrease in brightness
subtracted = cv2.subtract(image,M)
cv2.imshow("Subtracted",subtracted)
cv2.waitKey()
cv2.destroyAllWindows()

#Bitwise operations and masking
#uint8 datatype for storing images in OpenCV
import cv2
import numpy as np
#Making a square
#if we are doing a colored image we'd use rectangle = np.zeros((300,300,3),np.uint8)
square = np.zeros((300,300),np.uint8)
cv2.rectangle(square,(50,50),(250,250),255,-2)
cv2.imshow("Square",square)

#Making an ellipse
ellipse = np.zeros((300,300),np.uint8)
cv2.ellipse(ellipse,(150,150),(150,150),30,0,180,255,-1)
cv2.imshow("Ellipse",ellipse)


#Show only where they intersect
And = cv2.bitwise_and(square,ellipse)
cv2.imshow("AND",And)
cv2.waitKey(0)


#Show where either square or ellipse is
bitwiseOr = cv2.bitwise_or(square,ellipse)
cv2.imshow("OR",bitwiseOr)
cv2.waitKey(0)

#Shows where either exists by itself
bitwiseXor = cv2.bitwise_xor(square,ellipse)
cv2.imshow("XOR",bitwiseXor)
cv2.waitKey(0)

#Shows everything that isn't part of the square
#Notice that this operation inverts the image totally
bitwiseNot_sq = cv2.bitwise_not(square)
cv2.imshow("NOT-SQUARE",bitwiseNot_sq)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Blur,Sharpening,Canny for edge detection etc etc


#CHAPTER 3
#Segmentation = Partitioning images into different regions
#STEP 1 - Grayscaling essential when finding contours(cv2.findContours doesn't work if the image is colored)
#STEP 2 - Find Canny Edges(not necessary, but it reduces a lot of noise in practice when finding contours )
#     2.1 - cv2.findContours modifies the original image, so use .copy() method to copy the initial image into a temp variable
#         - cv2.findContours returns the contours and the hierarchy
#STEP 3 - drawContours(image,contours,nb_of_contours_to_draw(-1 if you draw thm all),(0,255,0)=color(green here),and last param = thickness of the contour)
import cv2
import numpy as np
#Step 1.1 = Loading the image
image = cv2.imread('image_to_detect_2.png')
cv2.imshow('Original Image',image)
cv2.waitKey(0)
#Step 1.2 = Converting the image to grayscale
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscaled Image',gray)
cv2.waitKey(0)
#Step 1.3 = Bilateral filtering
#Best for removing noise but still keeping the edges
bilateral = cv2.bilateralFilter(gray,3,75,75)
cv2.imshow('Bilateral Blurring',bilateral)
cv2.waitKey(0)
#Step 2.0 = Applying the Canny Edge Detection Algorithm
#Find Canny Edges
edged = cv2.Canny(bilateral,50,120)
cv2.imshow("Canny Edges",edged)
cv2.waitKey(0)
#Step 2.1 = Finding contours, use a copy of your image since findContours alters the image
_,contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("Number of Contours found = " + str(len(contours)))
cv2.imshow("Canny Edge After Contouring",edged)
cv2.waitKey(0)
#Step 3 - Sorting the contours
sorted_contours = sorted(contours,key=cv2.contourArea,reverse=True)
#Step 4 - Drawing the contours
"""
count = 0
for c in sorted_contours:
    cv2.drawContours(image,[c],-1,(0,255,0),2)
    if(count >= 50):
        break
    count+=1
"""
#Drawing all contours
cv2.drawContours(image,contours,-1,(0,255,0),1)
cv2.imshow('Contours',image)


#LICENTA - SOLUTION 1 --- STILL HAVE TO WORK ON IT
#Step 5 - Load the template or reference image
import cv2
template = cv2.imread('Circle.PNG')
target = cv2.imread('image_to_detect_2.png')
target_grayscaled = cv2.cvtColor(target,cv2.COLOR_BGR2GRAY)
circle_template_grayscaled = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
cv2.imshow('Template image',template)
cv2.waitKey(0)
cv2.imshow('Template image grayscaled',circle_template_grayscaled)
cv2.waitKey(0)
cv2.imshow('Initial Image',target)
cv2.waitKey(0)
cv2.imshow('Initial Image grayscaled',target_grayscaled)
cv2.waitKey(0)



ret,thresh1 = cv2.threshold(circle_template_grayscaled,127,255,0)
ret,thresh2 = cv2.threshold(target_grayscaled,127,255,0)

#edged_circle_template = cv2.Canny(circle_template_grayscaled,50,120)
#cv2.imshow("Circle Template After Applying Canny Edges",edged_circle_template)
#cv2.waitKey(0)


#Finding contours in the template
_,contours,hierarchy = cv2.findContours(thresh1,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)

#We need to sort the contours by area so that we can remove the largest
#contour which is image outline
sorted_contours = sorted(contours,key=cv2.contourArea,reverse=True)


#We extract the second largest contour which will be our template contour
template_contour = contours[1]

#Draw the contour in the initial circle image
cv2.drawContours(template,template_contour,-1,(0,255,0),3)
cv2.imshow('Initial Circle Template Contoured',template)
cv2.waitKey(0)


#Finding contours in the target
target_edged = cv2.Canny(target_grayscaled,50,120)
cv2.imshow("Target Image after applying Canny Edge",target_edged)
cv2.waitKey(0)


#Extract the contours from the target image
_,contours,hierarchy = cv2.findContours(target_edged,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
print(len(contours))

#Iterate through each contour in the target image
#Use cv2.matchShapes to compare contour shapes
#The smaller the better
matches_list = []
for c in contours:
    match = cv2.matchShapes(template_contour,c,1,0.0)
    matches_list.append(match)


matched_contour = min(matches_list)
contour_index = matches_list.index(matched_contour)


print("Min value is:" , matched_contour)
print("Min value position(in initial list) is:", contour_index)
cv2.drawContours(target,contours,contour_index,(0,255,0),3)
cv2.imshow('Output',target)
cv2.waitKey(0)
cv2.destroyAllWindows()





#LICENTA  - SOLUTON 2
import cv2
import numpy as np
#target = cv2.imread('image_to_detect_2.png')
#target = cv2.imread('multiple_red.jpg')
#target = cv2.imread('multiple_diff.jpg')
#target = cv2.imread('multiple_diff_2.jpg')
#target = cv2.imread('stop_sign.jpg')
#Arata ca aici se mai intampla sa iti deseneze is o parte foarte mica de pixeli
#target = cv2.imread('pietoni+stop.jpg')
target = cv2.imread('semn_multiplu.jpg')
#Asta nu merge intentionat pt ca scrie peste ea, demonstreaza la licenta ca daca ai noise nu merge bine
#target = cv2.imread('TwoBlueSigns.jpg')
target = cv2.resize(target,(700,800),interpolation=cv2.INTER_AREA)
print(target.shape)
target_hsv = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)
cv2.imshow('Initial Image',target)
cv2.waitKey(0)
cv2.imshow('HSV Image',target_hsv)
cv2.waitKey(0)

###############################TEMPLATES AND CONTOURS###############################################
#Circle Template&Contour
def retrieveCircleContour():
    circle_template = cv2.imread('Circle.PNG')
    circle_template_grayscaled = cv2.cvtColor(circle_template,cv2.COLOR_BGR2GRAY)
    ret,thresh1 = cv2.threshold(circle_template_grayscaled,127,255,0)
    _,contours,hierarchy = cv2.findContours(thresh1,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    sorted(contours,key=cv2.contourArea,reverse=True)
    circle_contour = contours[1]
    return circle_contour

#Triangle Template&Contour
def retrieveTriangleContour():
    triangle_template = cv2.imread('Triangle.PNG')
    triangle_template_grayscaled = cv2.cvtColor(triangle_template,cv2.COLOR_BGR2GRAY)
    ret,thresh1 = cv2.threshold(triangle_template_grayscaled,127,255,0)
    _,contours,hierarchy = cv2.findContours(thresh1,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    sorted(contours,key=cv2.contourArea,reverse=True)
    triangle_contour = contours[1]
    return triangle_contour

#Square Template&Contour
def retrieveSquareContour():
    square_template = cv2.imread('Square.PNG')
    square_template_grayscaled = cv2.cvtColor(square_template,cv2.COLOR_BGR2GRAY)
    ret,thresh1 = cv2.threshold(square_template_grayscaled,127,255,0)
    _,contours,hierarchy = cv2.findContours(thresh1,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    sorted(contours,key=cv2.contourArea,reverse=True)
    square_contour = contours[1]
    return square_contour

#Hexagon Template&Contour
def retrieveHexagonContour():
    hexagon_template = cv2.imread('Hexagon.JPG')
    hexagon_template_grayscaled = cv2.cvtColor(hexagon_template,cv2.COLOR_BGR2GRAY)
    ret,thresh1 = cv2.threshold(hexagon_template_grayscaled,127,255,0)
    _,contours,hierarchy = cv2.findContours(thresh1,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    sorted(contours,key=cv2.contourArea,reverse=True)
    hexagon_contour = contours[0]
    return hexagon_contour

#Oval Template&Contour
def retrieveOvalContour():
    oval_template = cv2.imread('Oval_Height.PNG')
    oval_template_grayscaled = cv2.cvtColor(oval_template,cv2.COLOR_BGR2GRAY)
    ret,thresh1 = cv2.threshold(oval_template_grayscaled,127,255,0)
    _,contours,hierarchy = cv2.findContours(thresh1,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    sorted(contours,key=cv2.contourArea,reverse=True)
    oval_height_contour = contours[0]
    return oval_height_contour

#Oval Template&Contour
def retrieveRhombusContour():
    rhombus_template = cv2.imread('Rhombus.PNG')
    rhombus_template_grayscaled = cv2.cvtColor(rhombus_template,cv2.COLOR_BGR2GRAY)
    ret,thresh1 = cv2.threshold(rhombus_template_grayscaled,127,255,0)
    _,contours,hierarchy = cv2.findContours(thresh1,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    sorted(contours,key=cv2.contourArea,reverse=True)
    rhombus_contour = contours[0]
    cv2.drawContours(rhombus_template,rhombus_contour,-1,(0,255,0),3)
    return rhombus_contour
   

#Red Mask
def retrieveRedMask(target_hsv):
    red_RGB = np.uint8([[[255,0,0]]])
    cv2.cvtColor(red_RGB,cv2.COLOR_BGR2HSV)
    red_mask_1 = cv2.inRange(target_hsv,np.array([170,100,100]),np.array([180,255,255]))
    red_mask_2 = cv2.inRange(target_hsv,np.array([0,100,100]), np.array([10,255,255]))
    red_mask = red_mask_1 | red_mask_2
    return red_mask

#Blue Mask
def retrieveBlueMask(target_hsv):
    blue_RGB = np.uint8([[[255,0,0]]])
    night_blue_RGB = np.uint8([[[37, 23, 35]]])
    cv2.cvtColor(blue_RGB,cv2.COLOR_BGR2HSV)
    lower_blue = np.array([110,150,50])
    upper_blue = np.array([130,255,255])
    day_blue_mask = cv2.inRange(target_hsv,lower_blue,upper_blue)
    
    
    night_blue_RGB = np.uint8([[[31, 17, 28]]])
    night_blue_HSV = cv2.cvtColor(night_blue_RGB,cv2.COLOR_BGR2HSV)
    night_blue_lower_bound = np.array([134,100,31])
    night_blue_upper_bound = np.array([154,255,255])
    night_blue_mask = cv2.inRange(target_hsv,night_blue_lower_bound,night_blue_upper_bound)
    blue_mask = day_blue_mask | night_blue_mask
    return blue_mask

#Yellow Mask
def retrieveYellowMask(target_hsv):
    lower_yellow = np.array([15,100, 100])
    upper_yellow = np.array([60, 255, 255])
    white_mask = cv2.inRange(target_hsv,lower_yellow,upper_yellow)
    return white_mask    

#Yellowish-Brownish Mask(for shadowed photos with priority sign)
def retrieveYellowishBrownishMask(target_hsv):
    #Yellowish-Brown
    yellowish_brownish = np.uint8([[[4, 46, 69]]])
    hsv_yellowish_brownish = cv2.cvtColor(yellowish_brownish,cv2.COLOR_BGR2HSV)
    lower_bound = np.array([9,100,60])
    upper_bound = np.array([29,255,120])
    yellowish_brownish_mask = cv2.inRange(target_hsv,lower_bound,upper_bound)
    return yellowish_brownish_mask



def applyClosingAndRetrieveMasks(red_mask,blue_mask,yellow_mask,yellowish_brownish_mask):
    morphological_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,ksize=(5,5))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, morphological_kernel);
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, morphological_kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, morphological_kernel)
    yellowish_brownish_mask = cv2.morphologyEx(yellowish_brownish_mask, cv2.MORPH_CLOSE, morphological_kernel)
    return red_mask,blue_mask,yellow_mask,yellowish_brownish_mask

#Display extractions
#Bitwise-AND mask and original image
def displayExtractions(target,red_mask,blue_mask,yellow_mask,yellowish_brownish_mask):
   # red_extraction = cv2.bitwise_and(target,target, mask=red_mask)
   # blue_extraction = cv2.bitwise_and(target,target, mask=blue_mask)
   # yellow_extraction = cv2.bitwise_and(target,target, mask=yellow_mask)
   # yellowish_brownish_extraction = cv2.bitwise_and(target,target,mask=yellowish_brownish_mask)
   # cv2.imshow('Red Extraction',red_extraction)
   # cv2.waitKey(0)
   # cv2.imshow('Blue Extraction',blue_extraction)
   # cv2.waitKey(0)
   # cv2.imshow('Yellow Extraction',yellow_extraction)
   # cv2.waitKey(0)
   # cv2.imshow('Dark-Yellow-Brownish Extraction',yellowish_brownish_extraction)
   # cv2.waitKey(0)
    return 

def sortAndRetrieveRedAndBlueContours(red_mask,blue_mask):
    _,red_contours,hierarchy = cv2.findContours(red_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    _,blue_contours,hierarchy = cv2.findContours(blue_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
   # print("Number of red contours found", len(red_contours))
   # print("Number of blue contours found",len(blue_contours))
    sorted_red_contours = sorted(red_contours,key=cv2.contourArea,reverse=True)
    sorted_blue_contours = sorted(blue_contours,key=cv2.contourArea,reverse=True)
    return sorted_red_contours,sorted_blue_contours

def sortAndRetrieveYellowAndYellowishAndBrownishContours(yellow_mask,yellowish_brownish_mask):
    _,yellow_contours,hierarchy = cv2.findContours(yellow_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    _,yellowish_brownish_contours,hierarchy = cv2.findContours(yellowish_brownish_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
   # print("Number of yellow contours found", len(yellow_contours))
   # print("Number of yellowish-brownish contours found",len(yellowish_brownish_contours))
    sorted_yellow_contours = sorted(yellow_contours,key=cv2.contourArea,reverse=True)
    sorted_yellowish_brownish_contours = sorted(yellowish_brownish_contours,key=cv2.contourArea,reverse=True)
    return sorted_yellow_contours,sorted_yellowish_brownish_contours


def retrieveContoursToBeDrawn(contours,color_flag,circle_contour,triangle_contour,square_contour,hexagon_contour,oval_contour):
    best_red_index = 0
    best_red_match = 1
    best_blue_index = 0
    best_blue_match = 1
    contours_to_be_drawn = []
    for (index,c) in enumerate(contours[:3]):
         match_circle = cv2.matchShapes(circle_contour,c,1,0.0)
         match_triangle = cv2.matchShapes(triangle_contour,c,1,0.0)
         match_square = cv2.matchShapes(square_contour,c,1,0.0)
         match_hexagon = cv2.matchShapes(hexagon_contour,c,1,0.0)
         match_oval = cv2.matchShapes(oval_contour,c,1,0.0)

       
         #Initialise Flags
         may_be_blue_circle = False
         may_be_blue_triangle = False
         may_be_blue_square = False
         may_be_blue_hexagon = False
         may_be_blue_oval = False
         may_be_blue_contours = False
         may_be_red_circle = False
         may_be_red_triangle = False
         may_be_red_square = False
         may_be_red_hexagon = False
         may_be_red_oval = False
         may_be_red_contours = False
         
         if(match_circle < 0.25): 
             if(color_flag == 0): 
                 may_be_blue_circle = True
                 may_be_blue_contours = True
                 if(match_circle < best_blue_match): 
                    best_blue_match = match_circle
                    best_blue_index = index
             else:
                 may_be_red_circle = True
                 may_be_red_contours = True
                 if(match_circle < best_red_match): 
                    best_red_match = match_circle
                    best_red_index = index
   
         if(match_triangle < 0.25):
             if(color_flag == 0):
                  may_be_blue_triangle = True
                  may_be_blue_contours = True
                  if(match_triangle < best_blue_match): 
                      best_blue_match = match_circle
                      best_blue_index = index
             else:
                  may_be_red_contours = True
                  may_be_red_triangle = True
                  if(match_triangle < best_red_match) : 
                    best_red_match = match_circle
                    best_red_index = index
     
                
         if(match_square < 0.25):
             if(color_flag == 0):
                 may_be_blue_square = True
                 may_be_blue_contours = True
                 if(match_square < best_blue_match): 
                    best_blue_match = match_square
                    best_blue_index = index
             else:
                 may_be_red_contours = True
                 may_be_red_square = True
                 if(match_square < best_red_match): 
                    best_red_match = match_square
                    best_red_index = index
      
                 
         if(match_hexagon < 0.25):
             if(color_flag == 0):
                 may_be_blue_hexagon = True
                 may_be_blue_contours = True
                 if(match_hexagon < best_blue_match): 
                    best_blue_match = match_hexagon
                    best_blue_index = index
             else:
                 may_be_red_contours = True
                 may_be_red_hexagon = True
                 if(match_square < best_red_match): 
                    best_red_match = match_hexagon
                    best_red_index = index
            
         if(match_oval < 0.25):
             if(color_flag == 0):
                 may_be_blue_oval = True
                 may_be_blue_contours = True
                 if(match_oval < best_blue_match): 
                    best_blue_match = match_oval
                    best_blue_index = index
             else:
                 may_be_red_contours = True
                 may_be_red_oval = True
                 if(match_square < best_red_match): 
                    best_red_match = match_oval
                    best_red_index = index
                  
         if(may_be_red_circle or may_be_red_hexagon or may_be_red_oval or may_be_red_triangle or may_be_red_square or may_be_blue_circle or may_be_blue_hexagon or may_be_blue_oval or may_be_blue_triangle or may_be_blue_square):
                contours_to_be_drawn.append(c)
    return contours_to_be_drawn


def retrieveYellowishContoursToBeDrawn(contours,color_flag,rhombus_contour):
    contours_to_be_drawn = []
    may_be_yellowish_rhombus = False
    may_be_yellowish_brownish_rhombus = False
    if(len(contours)>0):     
      for (index,c) in enumerate(contours[:1]):
         rhombus_match = cv2.matchShapes(rhombus_contour,c,1,0.0)
         if(rhombus_match < 0.4): 
             if(color_flag == 0): 
                 may_be_yellowish_rhombus = True
             else:
                 may_be_yellowish_brownish_rhombus = True
    if(may_be_yellowish_rhombus or may_be_yellowish_brownish_rhombus):
              contours_to_be_drawn.append(c)
    return contours_to_be_drawn


def makeUniquePrediction(image_path):
        initialTime = dt.datetime.now()
        test_image = image.load_img(image_path,target_size = (64,64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = loaded_model.predict(test_image)
        finishTime = dt.datetime.now()
        forwardPassTime = finishTime - initialTime
        print('Total prediction time: ', forwardPassTime)       
        flattened_result = [item for sublist in result for item in sublist]
        max_index, max_value = max(enumerate(flattened_result), key=operator.itemgetter(1))
       # print('Max index is:',max_index, ',Max value is:',max_value) 
        return max_index

def drawContoursAndMakePrediction(target,contours,contour_type):
 #DE CE PLM NU FACE DIRECT PREDICTIA SI TREBUIE SA REINCARCE DE PE DISC WTF WTF WTF?
 #RAMANE BLOCAT PE PRIMA PREDICITIE SI NUMA LA AIA RAMANE N-ARE SENS WTF?PYTHON REFERENCE?
   for i in range(0,len(contours)):
         hull = cv2.convexHull(contours[i],returnPoints = True)
         x,y,w,h = cv2.boundingRect(hull)
         result = ""
         shouldDrawContour = False
         if(contour_type == 0):
               cv2.rectangle(target,(x,y),(x+w,y+h),(0,255,0),2)
               cropped = target[y:y+h, x:x+w]
               #print('Width is: ' ,w , 'Height is: ',h)
               if(cropped.shape[0] > 64) and (cropped.shape[1] > 64):
                   resized_crop = cv2.resize(cropped,(64,64),interpolation = cv2.INTER_AREA)
                   image_name = 'Resized_crop_'+str(i)+'.jpg'
                 #  cv2.imshow('Cropped',resized_crop)
                 #  cv2.waitKey(0)
                   cv2.imwrite(image_name,resized_crop)
                   shouldDrawContour = True
                   result = makeUniquePrediction(image_name)
                   
         else:
               cv2.rectangle(target,(x-int(w/3),y-int(h/3)),(x+w+int(w/3),y+h+int(h/3)),(0,255,0),2)
               cropped = target[y-int(h/3):y+h+int(h/3), x-int(w/3):x+w+int(w/3)]
               #print('Width is: ' ,w , 'Height is: ',h)
               if(cropped.shape[0] > 64 ) and (cropped.shape[1] > 64):
                   resized_crop = cv2.resize(cropped,(64,64),interpolation = cv2.INTER_AREA)
                   image_name = 'Resized_crop_yellow_0'+str(i)+'.jpg'
                  # cv2.imshow('Cropped',resized_crop)
                  # cv2.waitKey(0)
                   cv2.imwrite(image_name,resized_crop)
                   shouldDrawContour = True
                   result = makeUniquePrediction(image_name)
                   
         for k,v in training_set.class_indices.items():
           if (v == result): 
               cv2.putText(target, k, (x-100,y+30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,20,147), 4) 
         if(shouldDrawContour):
             cv2.drawContours(target,[hull],-1,(255,20,147),2)
   return
 
    
import cv2
import numpy as np
import datetime as dt
import operator
from cnn import importDataSets
from cnn import retrieveModel
from keras.preprocessing import image
training_set,_,_ = importDataSets()
loaded_model = retrieveModel()
circle_contour = retrieveCircleContour()
triangle_contour = retrieveTriangleContour()
square_contour = retrieveSquareContour()
hexagon_contour = retrieveHexagonContour()
oval_contour = retrieveOvalContour()
rhombus_contour = retrieveRhombusContour()

initialTime = dt.datetime.now()
target = cv2.imread('multiple_ger_3.jpg')
target = cv2.resize(target,(700,800),interpolation=cv2.INTER_AREA)
initialTime = dt.datetime.now()
target_hsv = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)
red_mask,blue_mask,yellow_mask,yellowish_brownish_mask = retrieveRedMask(target_hsv),retrieveBlueMask(target_hsv),retrieveYellowMask(target_hsv),retrieveYellowishBrownishMask(target_hsv)
red_mask,blue_mask,yellow_mask,yellowish_brownish_mask = applyClosingAndRetrieveMasks(red_mask,blue_mask,yellow_mask,yellowish_brownish_mask)
#displayExtractions(target,red_mask,blue_mask,yellow_mask,yellowish_brownish_mask)
sorted_red_contours,sorted_blue_contours = sortAndRetrieveRedAndBlueContours(red_mask.copy(),blue_mask.copy())
sorted_yellow_contours,sorted_yellowish_brownish_contours = sortAndRetrieveYellowAndYellowishAndBrownishContours(yellow_mask.copy(),yellowish_brownish_mask.copy())
red_contours_to_be_drawn = retrieveContoursToBeDrawn(sorted_red_contours,1,circle_contour,triangle_contour,square_contour,hexagon_contour,oval_contour)
blue_contours_to_be_drawn = retrieveContoursToBeDrawn(sorted_blue_contours,0,circle_contour,triangle_contour,square_contour,hexagon_contour,oval_contour)
yellow_contours_to_be_drawn = retrieveYellowishContoursToBeDrawn(sorted_yellow_contours,0,rhombus_contour)
yellowish_brownish_contours_to_be_drawn =  retrieveYellowishContoursToBeDrawn(sorted_yellowish_brownish_contours,1,rhombus_contour)
if(len(red_contours_to_be_drawn) > 0):  
   drawContoursAndMakePrediction(target,red_contours_to_be_drawn,0)
if(len(blue_contours_to_be_drawn) > 0):  
   drawContoursAndMakePrediction(target,blue_contours_to_be_drawn,0)
if(len(yellow_contours_to_be_drawn) > 0):  
   drawContoursAndMakePrediction(target,yellow_contours_to_be_drawn,1)
if(len(yellowish_brownish_contours_to_be_drawn) > 0):  
   drawContoursAndMakePrediction(target,yellowish_brownish_contours_to_be_drawn,1)
finishTime = dt.datetime.now()
forwardPassTime = finishTime - initialTime
print('Detection + Classification Time',forwardPassTime) 
cv2.imshow('Final Image',target)
cv2.waitKey()
cv2.destroyAllWindows()




#NIGHT TIME ADAPTATION
def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
    
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

import cv2
import numpy as np
target = cv2.imread('noapte_3.jpg')
target = cv2.resize(target,(700,800),interpolation=cv2.INTER_AREA)
for gamma in np.arange(0.0, 3.5, 0.5):
	# ignore when gamma is 1 (there will be no change to the image)
	if gamma == 1:
		continue
	# apply gamma correction and show the images
	gamma = gamma if gamma > 0 else 0.1
	adjusted = adjust_gamma(target, gamma=gamma)
	cv2.putText(adjusted, "g={}".format(gamma), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
	cv2.imshow("Images", np.hstack([target, adjusted]))  
	cv2.waitKey(0)

#TEMPLATE MATCHING - CROSS CORRELATION
#Doesn't work, demonstreaza
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('prioritate.jpg',0)
img2 = img.copy()
template = cv2.imread('priority_template.png',0)
img = cv2.resize(img,(700,800),interpolation=cv2.INTER_AREA)
template = cv2.resize(template,(100,100),interpolation=cv2.INTER_AREA)
w, h = template.shape[::-1]

# All the 6 methods for comparison in a list
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for meth in methods:
    img = img2.copy()
    img = cv2.resize(img,(700,800),interpolation=cv2.INTER_AREA)
    method = eval(meth)

    # Apply template Matching
    res = cv2.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img,top_left, bottom_right, 255, 2)
    cv2.imshow("Result",img)
    cv2.waitKey(0)
"""






