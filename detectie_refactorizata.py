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
    blue_RGB = np.uint8([[[0,0,255]]])
    cv2.cvtColor(blue_RGB,cv2.COLOR_BGR2HSV)
    lower_blue = np.array([110,150,50])
    upper_blue = np.array([130,255,255])
    blue_mask = cv2.inRange(target_hsv,lower_blue,upper_blue)
    return blue_mask

def applyClosingAndRetrieveMasks(red_mask,blue_mask):
    morphological_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,ksize=(5,5))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, morphological_kernel);
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, morphological_kernel)
    return red_mask,blue_mask

#Display extractions
#Bitwise-AND mask and original image
def displayExtractions(target,red_mask,blue_mask):
    red_extraction = cv2.bitwise_and(target,target, mask=red_mask)
    blue_extraction = cv2.bitwise_and(target,target, mask=blue_mask)
    cv2.imshow('Red Extraction',red_extraction)
    cv2.imshow('Blue Extraction',blue_extraction)
    cv2.waitKey(0)
    return 

def sortAndRetrieveRedAndBlueContours(red_mask,blue_mask):
    _,red_contours,hierarchy = cv2.findContours(red_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    _,blue_contours,hierarchy = cv2.findContours(blue_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    print("Number of red contours found", len(red_contours))
    print("Number of blue contours found",len(blue_contours))
    sorted_red_contours = sorted(red_contours,key=cv2.contourArea,reverse=True)
    sorted_blue_contours = sorted(blue_contours,key=cv2.contourArea,reverse=True)
    return sorted_red_contours,sorted_blue_contours





def retrieveContoursToBeDrawn(contours,color_flag,circle_contour,triangle_contour,square_contour,hexagon_contour,oval_contour):
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
    best_red_index = 0
    best_red_match = 1
    best_blue_index = 0
    best_blue_match = 1
    contours_to_be_drawn = []
    for (index,c) in enumerate(contours[:8]):
     match_circle = cv2.matchShapes(circle_contour,c,1,0.0)
     match_triangle = cv2.matchShapes(triangle_contour,c,1,0.0)
     match_square = cv2.matchShapes(square_contour,c,1,0.0)
     match_hexagon = cv2.matchShapes(hexagon_contour,c,1,0.0)
     match_oval = cv2.matchShapes(oval_contour,c,1,0.0)
     print('Circle Match',match_circle)
     print('Triangle Match',match_triangle)
     print('Square Match',match_square)
     print('Hexagon Match',match_hexagon)
     print('Oval Match',match_oval)
     if(match_circle < 0.3): 
         if(color_flag == 0): 
             may_be_blue_circle = True
             may_be_blue_contours = True
         else:
             may_be_red_circle = True
             may_be_red_contours = True
         contours_to_be_drawn.append(c)
         if(color_flag == 0):
            if(match_circle < best_blue_match) : 
                best_blue_match = match_circle
                best_blue_index = index
         else:
            if(match_circle < best_red_match) : 
                best_red_match = match_circle
                best_red_index = index
            
   
       
     if(match_triangle < 0.3):
         if(color_flag == 0):
              may_be_blue_triangle = True
              may_be_blue_contours = True
         else:
              may_be_red_contours = True
              may_be_red_triangle = True
         contours_to_be_drawn.append(c)
         if(color_flag == 0):
            if(match_triangle < best_blue_match) : 
                best_blue_match = match_circle
                best_blue_index = index
         else:
            if(match_triangle < best_red_match) : 
                best_red_match = match_circle
                best_red_index = index

            
     if(match_square < 0.3):
         if(color_flag == 0):
             may_be_blue_square = True
             may_be_blue_contours = True
         else:
             may_be_red_contours = True
             may_be_red_square = True
         contours_to_be_drawn.append(c)
         if(color_flag == 0):
            if(match_square < best_blue_match) : 
                best_blue_match = match_square
                best_blue_index = index
         else:
            if(match_square < best_red_match) : 
                best_red_match = match_square
                best_red_index = index
  
             
     if(match_hexagon < 0.3):
         if(color_flag == 0):
             may_be_blue_hexagon = True
             may_be_blue_contours = True
         else:
             may_be_red_contours = True
             may_be_red_hexagon = True
         contours_to_be_drawn.append(c)
         if(color_flag == 0):
            if(match_hexagon < best_blue_match) : 
                best_blue_match = match_hexagon
                best_blue_index = index
         else:
            if(match_square < best_red_match) : 
                best_red_match = match_hexagon
                best_red_index = index
        
     if(match_oval < 0.3):
         if(color_flag == 0):
             may_be_blue_oval = True
             may_be_blue_contours = True
         else:
             may_be_red_contours = True
             may_be_red_oval = True
         contours_to_be_drawn.append(c)
         if(color_flag == 0):
            if(match_oval < best_blue_match) : 
                best_blue_match = match_oval
                best_blue_index = index
         else:
            if(match_square < best_red_match) : 
                best_red_match = match_oval
                best_red_index = index
    return contours_to_be_drawn

def drawContours(target,contours):
   for i in range(0,len(contours)):
         hull = cv2.convexHull(contours[i],returnPoints = True)
         cv2.drawContours(target,[hull],-1,(0,255,0),2)
   return
###################### LOADING THE TEMPLATES ######################################


#cv2.drawContours(circle_template,circle_contour,-1,(0,255,0),3)
#cv2.imshow('Initial Circle Template Contoured',circle_template)
#cv2.waitKey(0)

#Triangle Template

#cv2.drawContours(triangle_template,triangle_contour,-1,(0,255,0),3)
#cv2.imshow('Initial Triangle Template Contoured',triangle_template)
#cv2.waitKey(0)

#Square Template

#cv2.drawContours(square_template,square_contour,-1,(0,255,0),3)
#cv2.imshow('Initial Square Template Contoured',square_template)
#cv2.waitKey(0)

#Hexagon Template

#cv2.imshow('Initial Hexagon Template Contoured',hexagon_template)
#cv2.waitKey(0)
# cv2.drawContours(hexagon_template,hexagon_contour,-1,(0,255,0),3)

#Oval Template
#Big In Height

#cv2.drawContours(oval_template,oval_height_contour,-1,(0,255,0),3)
#cv2.imshow('Initial Oval Height Template Contoured',oval_template)
#cv2.waitKey(0)
#Oval Template
#Big in Width



######################## USING MASKS TO FIND THE ROI #####################################
#Find masks for different colors
#Convert white from RGB to white in HSV
#To find the masks, [H-10, 100,100] and [H+10, 255, 255]
#Not completely true, check stackoverflow for errors/problems

#White Mask Image
#white_RGB = np.uint8([[[255,255,255]]])
#white_HSV = cv2.cvtColor(white_RGB,cv2.COLOR_BGR2HSV)
#white_mask = cv2.inRange(target_hsv,np.array([100,100,200]),np.array([255,255,255]))
#cv2.imshow('White Mask Image',white_mask)
#cv2.waitKey(0)

#Red Mask Image


#Blue Mask Image

#cv2.imshow('Blue Mask Image',blue_mask)
#cv2.waitKey(0)

#Apply morphological close
    

import cv2
import numpy as np
target = cv2.imread('multiple_red.jpg')
target = cv2.resize(target,(700,800),interpolation=cv2.INTER_AREA)
target_hsv = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)
circle_contour = retrieveCircleContour()
triangle_contour = retrieveTriangleContour()
square_contour = retrieveSquareContour()
hexagon_contour = retrieveHexagonContour()
oval_contour = retrieveOvalContour()
red_mask,blue_mask = retrieveRedMask(target_hsv),retrieveBlueMask(target_hsv)
red_mask,blue_mask = applyClosingAndRetrieveMasks(red_mask,blue_mask)
displayExtractions(target,red_mask,blue_mask)
sorted_red_contours,sorted_blue_contours = sortAndRetrieveRedAndBlueContours(red_mask.copy(),blue_mask.copy())
red_contours_to_be_drawn = retrieveContoursToBeDrawn(sorted_red_contours,1,circle_contour,triangle_contour,square_contour,hexagon_contour,oval_contour)
blue_contours_to_be_drawn = retrieveContoursToBeDrawn(sorted_blue_contours,0,circle_contour,triangle_contour,square_contour,hexagon_contour,oval_contour)
drawContours(target,red_contours_to_be_drawn)
drawContours(target,blue_contours_to_be_drawn)
cv2.imshow('Final Image',target)
cv2.waitKey()
cv2.destroyAllWindows()

