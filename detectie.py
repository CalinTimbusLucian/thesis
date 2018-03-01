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
target_width = target.shape[0]
target_height = target.shape[1]
target = cv2.resize(target,(700,800),interpolation=cv2.INTER_AREA)
print(target.shape)
target_hsv = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)
cv2.imshow('Initial Image',target)
cv2.waitKey(0)
cv2.imshow('HSV Image',target_hsv)
cv2.waitKey(0)



###################### LOADING THE TEMPLATES ######################################
#Circle Template
circle_template = cv2.imread('Circle.PNG')
circle_template_grayscaled = cv2.cvtColor(circle_template,cv2.COLOR_BGR2GRAY)
ret,thresh1 = cv2.threshold(circle_template_grayscaled,127,255,0)
_,contours,hierarchy = cv2.findContours(thresh1,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
sorted_contours = sorted(contours,key=cv2.contourArea,reverse=True)
circle_contour = contours[1]
cv2.drawContours(circle_template,circle_contour,-1,(0,255,0),3)
#cv2.imshow('Initial Circle Template Contoured',circle_template)
#cv2.waitKey(0)

#Triangle Template
triangle_template = cv2.imread('Triangle.PNG')
triangle_template_grayscaled = cv2.cvtColor(triangle_template,cv2.COLOR_BGR2GRAY)
ret,thresh1 = cv2.threshold(triangle_template_grayscaled,127,255,0)
_,contours,hierarchy = cv2.findContours(thresh1,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
sorted_contours = sorted(contours,key=cv2.contourArea,reverse=True)
triangle_contour = contours[1]
cv2.drawContours(triangle_template,triangle_contour,-1,(0,255,0),3)
#cv2.imshow('Initial Triangle Template Contoured',triangle_template)
#cv2.waitKey(0)

#Square Template
square_template = cv2.imread('Square.PNG')
square_template_grayscaled = cv2.cvtColor(square_template,cv2.COLOR_BGR2GRAY)
ret,thresh1 = cv2.threshold(square_template_grayscaled,127,255,0)
_,contours,hierarchy = cv2.findContours(thresh1,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
sorted_contours = sorted(contours,key=cv2.contourArea,reverse=True)
square_contour = contours[1]
cv2.drawContours(square_template,square_contour,-1,(0,255,0),3)
#cv2.imshow('Initial Square Template Contoured',square_template)
#cv2.waitKey(0)

#Hexagon Template
hexagon_template = cv2.imread('Hexagon.JPG')
hexagon_template_grayscaled = cv2.cvtColor(hexagon_template,cv2.COLOR_BGR2GRAY)
ret,thresh1 = cv2.threshold(hexagon_template_grayscaled,127,255,0)
_,contours,hierarchy = cv2.findContours(thresh1,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
sorted_contours = sorted(contours,key=cv2.contourArea,reverse=True)
hexagon_contour = contours[0]
cv2.drawContours(hexagon_template,hexagon_contour,-1,(0,255,0),3)
#cv2.imshow('Initial Hexagon Template Contoured',hexagon_template)
#cv2.waitKey(0)

#Oval Template
#Big In Height
oval_template = cv2.imread('Oval_Height.PNG')
oval_template_grayscaled = cv2.cvtColor(oval_template,cv2.COLOR_BGR2GRAY)
ret,thresh1 = cv2.threshold(oval_template_grayscaled,127,255,0)
_,contours,hierarchy = cv2.findContours(thresh1,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
sorted_contours = sorted(contours,key=cv2.contourArea,reverse=True)
oval_height_contour = contours[0]
cv2.drawContours(oval_template,oval_height_contour,-1,(0,255,0),3)
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
red_RGB = np.uint8([[[255,0,0]]])
red_HSV = cv2.cvtColor(red_RGB,cv2.COLOR_BGR2HSV)
red_mask_1 = cv2.inRange(target_hsv,np.array([170,100,100]),np.array([180,255,255]))
red_mask_2 = cv2.inRange(target_hsv,np.array([0,100,100]), np.array([10,255,255]))
red_mask = red_mask_1 | red_mask_2
cv2.imshow('Red Mask Image',red_mask)
cv2.waitKey(0)

#Blue Mask Image
blue_RGB = np.uint8([[[0,0,255]]])
blue_HSV = cv2.cvtColor(blue_RGB,cv2.COLOR_BGR2HSV)
lower_blue = np.array([110,150,50])
upper_blue = np.array([130,255,255])
blue_mask = cv2.inRange(target_hsv,lower_blue,upper_blue)
cv2.imshow('Blue Mask Image',blue_mask)
cv2.waitKey(0)

#Apply morphological close
morphological_kernel_for_red = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,ksize=(5,5))
morphological_kernel_for_blue = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,ksize=(5,5))
red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, morphological_kernel_for_red);
blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, morphological_kernel_for_blue)

# Bitwise-AND mask and original image
red_extraction = cv2.bitwise_and(target,target, mask=red_mask)
blue_extraction = cv2.bitwise_and(target,target, mask=blue_mask)
cv2.imshow('Red Extraction',red_extraction)
cv2.imshow('Blue Extraction',blue_extraction)
cv2.waitKey(0)


########################## FINAL PART = MATCHING SHAPES + CONTOUR COLORS ####################################
#Check the image to see if there are any contours to match the square/triangle/circle
#Randomly set it to a low threshold, if I don't find any thresholds lower than that value then I don't draw it <3
#Vezi la licenta arata ca fara match shapes imi deseneaza si o parte din drum(vede ca si bleumarin)
#Vezi ca am facut oval ca altfel imi detecta ceva punct rosu la distanta
_,red_contours,hierarchy = cv2.findContours(red_mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
_,blue_contours,hierarchy = cv2.findContours(blue_mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
print("Number of red contours found", len(red_contours))
print("Number of blue contours found",len(blue_contours))
sorted_red_contours = sorted(red_contours,key=cv2.contourArea,reverse=True)
sorted_blue_contours = sorted(blue_contours,key=cv2.contourArea,reverse=True)
may_be_blue_circle = False
may_be_blue_triangle = False
may_be_blue_square = False
may_be_blue_hexagon = False
may_be_blue_oval = False
may_be_red_circle = False
may_be_red_triangle = False
may_be_red_square = False
may_be_red_hexagon = False
may_be_red_oval = False
best_red_index = 0
best_red_match = 1
best_blue_index = 0
best_blue_match = 1
red_contours_to_be_drawn = []
blue_contours_to_be_drawn = []
for (index,c) in enumerate(sorted_blue_contours[:8]):
     match_circle = cv2.matchShapes(circle_contour,c,1,0.0)
     match_triangle = cv2.matchShapes(triangle_contour,c,1,0.0)
     match_square = cv2.matchShapes(square_contour,c,1,0.0)
     match_hexagon = cv2.matchShapes(hexagon_contour,c,1,0.0)
     match_oval = cv2.matchShapes(oval_height_contour,c,1,0.0)
     print('Blue Circle Match',match_circle)
     print('Blue Triangle Match',match_triangle)
     print('Blue Square Match',match_square)
     print('Blue Hexagon Match',match_hexagon)
     print('Blue Oval Match',match_oval)
     if(match_circle < 0.3): 
         may_be_blue_circle = True
         blue_contours_to_be_drawn.append(c)
         if(match_circle < best_blue_match) : 
             best_blue_match = match_circle
             best_blue_index = index
   
       
     if(match_triangle < 0.3): 
         may_be_blue_triangle = True
         blue_contours_to_be_drawn.append(c)
         if(match_triangle < best_blue_match):
             best_blue_match = match_triangle
             best_blue_index = index

            
     if(match_square < 0.3):
         may_be_blue_square = True
         blue_contours_to_be_drawn.append(c)
         if(match_square < best_blue_match):
             best_blue_match = match_square
             best_blue_index = index
  
             
     if(match_hexagon < 0.3):
         may_be_blue_hexagon = True
         blue_contours_to_be_drawn.append(c)
         if(match_hexagon < best_blue_match):
             best_blue_match = match_hexagon
             best_blue_index = index

        
     if(match_oval < 0.3):
         may_be_blue_oval = True
         blue_contours_to_be_drawn.append(c)
         if(match_oval < best_blue_match):
             best_blue_match = match_oval
             best_blue_index = index
        
   
for (index,c) in enumerate(sorted_red_contours[:8]):
     match_circle = cv2.matchShapes(circle_contour,c,1,0.0)
     match_triangle = cv2.matchShapes(triangle_contour,c,1,0.0)
     match_square = cv2.matchShapes(square_contour,c,1,0.0)
     match_hexagon = cv2.matchShapes(hexagon_contour,c,1,0.0)
     match_oval = cv2.matchShapes(oval_height_contour,c,1,0.0)
     print('Red Circle Match',match_circle)
     print('Red Triangle Match',match_triangle)
     print('Red Square Match',match_square)
     print('Red Hexagon Match',match_hexagon)
     print('Red Oval Match',match_oval)
     if(match_circle < 0.3): 
         may_be_red_circle = True
         red_contours_to_be_drawn.append(c)
         if(match_circle < best_red_match):
                 best_red_match = match_circle
                 best_red_index = index
  
     if(match_triangle < 0.3): 
         may_be_red_triangle = True
         red_contours_to_be_drawn.append(c)
         if(match_triangle < best_red_match):
                 best_red_match = match_triangle
                 best_red_index = index
        
     if(match_square < 0.3):
         may_be_red_square = True
         red_contours_to_be_drawn.append(c)
         if(match_circle < best_red_match):
                 best_red_match = match_square
                 best_red_index = index
                     
         
     if(match_hexagon < 0.3):
         may_be_red_hexagon = True
         red_contours_to_be_drawn.append(c)
         if(match_circle < best_red_match):
                 best_red_match = match_hexagon
                 best_red_index = index
                 
     if(match_oval < 0.3):
         may_be_red_oval = True
         red_contours_to_be_drawn.append(c)
         if(match_oval < best_red_match):
                 best_red_match = match_oval
                 best_red_index = index
            

print('Best Red Index is',best_red_index)
print('Best Red Match is',best_red_match)
print('Best Blue Index is',best_blue_index)
print('Best Blue Match is',best_blue_match)
if(may_be_red_circle or may_be_red_square or may_be_red_triangle or may_be_red_hexagon or may_be_red_oval):
    for i in range(0,len(red_contours_to_be_drawn)):
         hull = cv2.convexHull(red_contours_to_be_drawn[i],returnPoints = True)
         cv2.drawContours(target,[hull],-1,(0,255,0),2)
if(may_be_blue_circle or may_be_blue_square or may_be_blue_triangle or may_be_blue_hexagon or may_be_blue_oval):
    for i in range(0,len(blue_contours_to_be_drawn)):
         hull = cv2.convexHull(blue_contours_to_be_drawn[i],returnPoints = True)
         cv2.drawContours(target,[hull],-1,(0,255,0),2)
cv2.imshow('Final Image',target)
cv2.waitKey()
cv2.destroyAllWindows()