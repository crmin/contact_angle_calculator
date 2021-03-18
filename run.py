import sys

import cv2 as cv
import numpy as np
from PIL import ImageFont, ImageDraw, Image

# img = cv.imread('sample.png')
img = cv.imread(sys.argv[1])
original_img = img

hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
# lower = np.array([-10, 100, 100], dtype = "uint8")
# upper = np.array([10, 255, 255], dtype = "uint8")

# *******************************************************
# # Set range for red color and
# # define mask
# red_lower = np.array([136, 87, 111], np.uint8)
# red_upper = np.array([180, 255, 255], np.uint8)
# red_mask = cv.inRange(hsvFrame, red_lower, red_upper)

# # Set range for green color and
# # define mask
# green_lower = np.array([25, 52, 72], np.uint8)
# green_upper = np.array([102, 255, 255], np.uint8)
# green_mask = cv.inRange(hsvFrame, green_lower, green_upper)

# # Set range for blue color and
# # define mask
# blue_lower = np.array([94, 80, 2], np.uint8)
# blue_upper = np.array([120, 255, 255], np.uint8)
# blue_mask = cv.inRange(hsvFrame, blue_lower, blue_upper)
# *******************************************************

lower = np.array([-10, 100, 100])
upper = np.array([10, 255, 255])
mask = cv.inRange(hsv, lower, upper)
img = cv.bitwise_and(img, img, mask=mask)
img = cv.bitwise_not(img)

# ################ HoughLines
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# edges = cv.Canny(gray,50,200)
# lines = cv.HoughLines(edges,1,np.pi/180,200)

# for idx, line in enumerate(lines):
#     print(idx)
#     rho,theta = line[0]
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))
#     cv.line(img,(x1,y1),(x2,y2),(0,255,255),2)
#     break

# cv.imshow("result", img)
# # cv.imshow("result", gray)
# cv.waitKey(0)

# ################ HoughLinesP
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray, 50, 150)
lines = cv.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

detected_lines = []

for idx, line in enumerate(lines):
    # if idx % 2:
    #     continue
    x1, y1, x2, y2 = line[0]
    detected_lines.append(line[0])
    # cv.line(original_img,(x1,y1),(x2,y2),(255, 255, 0),2)

print('detected_lines:', detected_lines)

len_detected_lines = len(detected_lines)
print('len of detected_lines:', len_detected_lines)
if len_detected_lines == 2:
    pass
elif len_detected_lines >= 3 and len_detected_lines <= 4:
    detected_lines = [each for idx, each in enumerate(detected_lines) if idx % 2 == 0]

for detected_line in detected_lines:
    x1, y1, x2, y2 = detected_line
    cv.line(original_img,(x1,y1),(x2,y2),(255, 255, 0),2)

for detected_line in detected_lines:
    detected_line_point = [str(point) for point in detected_line]
    print(', '.join(detected_line_point))

((x1, y1, x2, y2), (x3, y3, x4, y4)) = detected_lines

# get angle
numerator = (x1 - x2) * (x3 - x4) + (y1 - y2) * (y3 - y4)
denominator = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) * np.sqrt((x3 - x4) ** 2 + (y3 - y4) ** 2)
angle = np.arccos(numerator / denominator)
angle_degree = np.rad2deg(angle)
angle_degree_left = 180 - angle_degree

print('angle:', angle_degree_left)

# font = cv.FONT_HERSHEY_SIMPLEX
# cv.putText(original_img, f'θ={angle_degree_left:.2f}', (10,40), font, 1, (255, 0, 255), 2, cv.LINE_AA)

img_pil = Image.fromarray(original_img)
draw = ImageDraw.Draw(img_pil)

fontpath = './d2coding.ttc'
font_bg = ImageFont.truetype(fontpath, 18)
draw.text((10, 10), f'θ={angle_degree_left:.2f}', font=font_bg, fill=(255, 255, 255))

font = ImageFont.truetype(fontpath, 18)
draw.text((10, 10), f'θ={angle_degree_left:.2f}', font=font, fill=(255, 0, 255))
original_img = np.array(img_pil)

cv.imshow("result", original_img)
cv.waitKey(0)