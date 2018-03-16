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
    cv2.cvtColor(blue_RGB,cv2.COLOR_BGR2HSV)
    lower_blue = np.array([110,150,50])
    upper_blue = np.array([130,255,255])
    blue_mask = cv2.inRange(target_hsv,lower_blue,upper_blue)
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
    red_extraction = cv2.bitwise_and(target,target, mask=red_mask)
    blue_extraction = cv2.bitwise_and(target,target, mask=blue_mask)
    yellow_extraction = cv2.bitwise_and(target,target,mask=yellow_mask)
    yellowish_brownish_extraction = cv2.bitwise_and(target,target,mask=yellowish_brownish_mask)
    cv2.imshow('Red Extraction',red_extraction)
    cv2.imshow('Blue Extraction',blue_extraction)
    cv2.imshow('Yellow Extraction',yellow_extraction)
    cv2.imshow('Dark-Yellow-Brownish Extraction',yellowish_brownish_extraction)
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

def sortAndRetrieveYellowAndYellowishAndBrownishContours(yellow_mask,yellowish_brownish_mask):
    _,yellow_contours,hierarchy = cv2.findContours(yellow_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    _,yellowish_brownish_contours,hierarchy = cv2.findContours(yellowish_brownish_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    print("Number of yellow contours found", len(yellow_contours))
    print("Number of yellowish-brownish contours found",len(yellowish_brownish_contours))
    sorted_yellow_contours = sorted(yellow_contours,key=cv2.contourArea,reverse=True)
    sorted_yellowish_brownish_contours = sorted(yellowish_brownish_contours,key=cv2.contourArea,reverse=True)
    return sorted_yellow_contours,sorted_yellowish_brownish_contours


def retrieveContoursToBeDrawn(contours,color_flag,circle_contour,triangle_contour,square_contour,hexagon_contour,oval_contour):
    best_red_index = 0
    best_red_match = 1
    best_blue_index = 0
    best_blue_match = 1
    contours_to_be_drawn = []
    for (index,c) in enumerate(contours[:5]):
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
         
         if(match_circle < 0.27): 
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
   
         if(match_triangle < 0.27):
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
     
                
         if(match_square < 0.27):
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
      
                 
         if(match_hexagon < 0.27):
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
            
         if(match_oval < 0.27):
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
    if(len(contours)>0):     
      for (index,c) in enumerate(contours[:2]):
         #Initialise Flags
         may_be_yellowish_rhombus = False
         may_be_yellowish_brownish_rhombus = False
         rhombus_match = cv2.matchShapes(rhombus_contour,c,1,0.0)
         if(rhombus_match < 0.4): 
             if(color_flag == 0): 
                 may_be_yellowish_rhombus = True
             else:
                 may_be_yellowish_brownish_rhombus = True
    if(may_be_yellowish_rhombus or may_be_yellowish_brownish_rhombus):
              contours_to_be_drawn.append(c)
    return contours_to_be_drawn



def retrieveAndDrawPossiblePrioritySign(target,rhombus_contour):
    best_match = 1
    best_index = 0
    for (index,c) in enumerate(sorted_contours[:2]):
        rhombus_match = cv2.matchShapes(rhombus_contour,c,1,0.0)
        if(rhombus_match < 0.4):
            print('Fit to be drawn')
            if(rhombus_match < best_match):
                 best_match = rhombus_match
                 best_index = index
    print('Best Match is', best_match)
    print('Best Index is', best_index)
    cv2.drawContours(target,sorted_contours[best_index],-1,(255,20,147),2)    
    cv2.imshow('Final Image',target)
    cv2.waitKey()
    return 


def drawContours(target,contours,contour_type):
  
   for i in range(0,len(contours)):
         hull = cv2.convexHull(contours[i],returnPoints = True)
         x,y,w,h = cv2.boundingRect(hull)
         if(contour_type == 0):
               cv2.rectangle(target,(x,y),(x+w,y+h),(0,255,0),2)
               cropped = target[y:y+h, x:x+w]
         else:
               print('Width is',w)
               print('Height is',h)
               cv2.rectangle(target,(x-int(w/3),y-int(h/3)),(x+w+int(w/3),y+h+int(h/3)),(0,255,0),2)
               cropped = target[y-int(h/3):y+h+int(h/3), x-int(w/3):x+w+int(w/3)]
         cv2.imshow('Cropped',cropped)
         image_name = 'Cropped'+str(i)+'.jpg'
         if(w>=30 and h>=30):
             cv2.imwrite(image_name,cropped)
         cv2.waitKey(0)
         cv2.drawContours(target,[hull],-1,(255,20,147),2)
   return

    
import cv2
import numpy as np
target = cv2.imread('multiple_diff.jpg')
target = cv2.resize(target,(700,800),interpolation=cv2.INTER_AREA)
target_hsv = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)
circle_contour = retrieveCircleContour()
triangle_contour = retrieveTriangleContour()
square_contour = retrieveSquareContour()
hexagon_contour = retrieveHexagonContour()
oval_contour = retrieveOvalContour()
rhombus_contour = retrieveRhombusContour()
red_mask,blue_mask,yellow_mask,yellowish_brownish_mask = retrieveRedMask(target_hsv),retrieveBlueMask(target_hsv),retrieveYellowMask(target_hsv),retrieveYellowishBrownishMask(target_hsv)
red_mask,blue_mask,yellow_mask,yellowish_brownish_mask = applyClosingAndRetrieveMasks(red_mask,blue_mask,yellow_mask,yellowish_brownish_mask)
displayExtractions(target,red_mask,blue_mask,yellow_mask,yellowish_brownish_mask)
sorted_red_contours,sorted_blue_contours = sortAndRetrieveRedAndBlueContours(red_mask.copy(),blue_mask.copy())
sorted_yellow_contours,sorted_yellowish_brownish_contours = sortAndRetrieveYellowAndYellowishAndBrownishContours(yellow_mask.copy(),yellowish_brownish_mask.copy())
red_contours_to_be_drawn = retrieveContoursToBeDrawn(sorted_red_contours,1,circle_contour,triangle_contour,square_contour,hexagon_contour,oval_contour)
blue_contours_to_be_drawn = retrieveContoursToBeDrawn(sorted_blue_contours,0,circle_contour,triangle_contour,square_contour,hexagon_contour,oval_contour)
yellow_contours_to_be_drawn = retrieveYellowishContoursToBeDrawn(sorted_yellow_contours,0,rhombus_contour)
yellowish_brownish_contours_to_be_drawn =  retrieveYellowishContoursToBeDrawn(sorted_yellowish_brownish_contours,1,rhombus_contour)
drawContours(target,red_contours_to_be_drawn,0)
drawContours(target,blue_contours_to_be_drawn,0)
drawContours(target,yellow_contours_to_be_drawn,1)
drawContours(target,yellowish_brownish_contours_to_be_drawn,1)
cv2.imshow('Final Image',target)
cv2.waitKey()
cv2.destroyAllWindows()


