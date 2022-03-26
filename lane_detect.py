import cv2
import numpy as np
import matplotlib.pyplot as plt


path='test_image.jpg'

img=cv2.imread(path)
#creating a copy of original image
lane_image=np.copy(img)

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1=image.shape[0]
    y2=int(y1*3/5)
    x1=int((y1-intercept)/slope)
    x2=int((y2-intercept)/slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit=[]
    right_fit=[]
    for line in lines:
        x1, y1, x2, y2=line.reshape(4)
        parameters=np.polyfit((x1,x2),(y1,y2),1)
        slope=parameters[0]
        intercept=parameters[1]
        if slope <0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])

def canny(image):
    #Converting it into gray_scale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #next step is to convert blur so image can be smmooth and easy for edge detection
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    # to detect edges
    canny=cv2.Canny(blur, 50,150)
    return canny

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines: 
            cv2.line(line_image, (x1,y1),(x2,y2), (255,0,0),10)
    return line_image


def region_of_intrest(image):
    height = image.shape[0]
    polyGon = np.array([
        [(200,height),(1100,height),(550,250)]
        ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polyGon, 255)
    masked_image=cv2.bitwise_and(image, mask)
    return masked_image


'''
# just calling the function
canny_img=canny(lane_image)
cropped_image=region_of_intrest(canny_img)
lines = cv2.HoughLinesP(cropped_image, 2,np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
average_lines=average_slope_intercept(lane_image, lines)

line_image = display_lines(lane_image,average_lines)

combo_img = cv2.addWeighted(lane_image, 0.8, line_image,1,1)
# showing image with the help of matplotlib

cv2.imshow('Lane',combo_img)
cv2.waitKey(0)


plt.imshow(canny)
plt.show()
'''

cap=cv2.VideoCapture('test2.mp4')
while(cap.isOpened()):
    _, frame=cap.read()
    canny_img=canny(frame)
    cropped_image=region_of_intrest(canny_img)
    lines = cv2.HoughLinesP(cropped_image, 2,np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    average_lines=average_slope_intercept(frame, lines)

    line_image = display_lines(frame,average_lines)

    combo_img = cv2.addWeighted(frame, 0.8, line_image,1,1)
# showing image with the help of matplotlib

    cv2.imshow('Lane',combo_img)
    
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()