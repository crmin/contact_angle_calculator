import sys

import cv2 as cv
import numpy as np
from PIL import ImageFont, ImageDraw, Image


img = cv.imread(sys.argv[1])
original_img = img

hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)

# @@@ 파란 원의 장축, 단축 좌표 계산
blue_lower = np.array([94, 80, 2], np.uint8)
blue_upper = np.array([120, 255, 255], np.uint8)
blue_mask = cv.inRange(hsv, blue_lower, blue_upper)
circle_img = cv.bitwise_and(img, img, mask=blue_mask)

shape_y, shape_x, _ = img.shape

"""
         (a)
       *     *
   *            *
 (b)             (c)
   *            *
       *     *
         (d)

(a): circle_top
(b): circle_left
(c): circle_right
(d): circle_bottom
"""

# find circle_top: 최상단에서 부터 내려오면서 [0, 0, ..., 0]이 아닌 위치 구하기
for y_idx in range(shape_y):
    if not np.array_equal(circle_img[y_idx], np.zeros((shape_x, 3))):
        x_arrays = []  # 최상단은 항상 한 점이 아닐 수 있기 때문에 연속된 점이 나온다면 중간 값을 취하도록 함
        for x_idx in range(shape_x):
            if not np.array_equal(circle_img[y_idx][x_idx], np.zeros(3)):
                x_arrays.append(x_idx)
        circle_top = (x_arrays[len(x_arrays)//2], y_idx)
        break

# find circle_bottom: 최하단에서 부터 내려오면서 [0, 0, ..., 0]이 아닌 위치 구하기
for y_idx in range(shape_y - 1, -1, -1):
    if not np.array_equal(circle_img[y_idx], np.zeros((shape_x, 3))):
        x_arrays = []  # 최상단은 항상 한 점이 아닐 수 있기 때문에 연속된 점이 나온다면 중간 값을 취하도록 함
        for x_idx in range(shape_x):
            if not np.array_equal(circle_img[y_idx][x_idx], np.zeros(3)):
                x_arrays.append(x_idx)
        circle_bottom = (x_arrays[len(x_arrays)//2], y_idx)
        break

# find circle_left: 왼쪽에서 부터 이동하면서 [0, 0, ..., 0]이 아닌 위치 구하기
for x_idx in range(shape_x):
    is_find = False
    y_arrays = []
    for y_idx in range(shape_y):
        if not np.array_equal(circle_img[y_idx][x_idx], np.zeros(3)):
            is_find = True
            y_arrays.append(y_idx)
    if is_find:
        circle_left = (x_idx, y_arrays[len(y_arrays)//2])
        break

# find circle_right: 오른쪽에서 부터 이동하면서 [0, 0, ..., 0]이 아닌 위치 구하기
for x_idx in range(shape_x - 1, -1, -1):
    is_find = False
    y_arrays = []
    for y_idx in range(shape_y):
        if not np.array_equal(circle_img[y_idx][x_idx], np.zeros(3)):
            is_find = True
            y_arrays.append(y_idx)
    if is_find:
        circle_right = (x_idx, y_arrays[len(y_arrays)//2])
        break

print('circle_top:', circle_top)
cv.circle(img, circle_top, 3, (0, 0, 255), -1)
print('circle_bottom:', circle_bottom)
cv.circle(img, circle_bottom, 3, (0, 255, 0), -1)
print('circle_left:', circle_left)
cv.circle(img, circle_left, 3, (255, 0, 0), -1)
print('circle_right:', circle_left)
cv.circle(img, circle_right, 3, (0, 255, 255), -1)


# @@@ 파란 원과 수평선의 접점 계산
# 수평선은 이론상 x축에 평행해야하지만 실제로는 다를 수 있으므로 왼쪽 끝과 오른쪽 끝 점의 y좌표를 구해서 평균으로 사용

red_lower = np.array([-10, 100, 100])
red_upper = np.array([10, 255, 255])
red_mask = cv.inRange(hsv, red_lower, red_upper)
line_img = cv.bitwise_and(img, img, mask=red_mask)
line_img = cv.bitwise_not(img)


gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray, 50, 150)
lines = cv.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

parallel_line = None

for idx, line in enumerate(lines):
    # if idx % 2:
    #     continue
    x1, y1, x2, y2 = line[0]
    parallel_line = line[0]
    break

# 선 인식 문제로 중간부터 그릴 경우 처음부터 그리기 위함
x_axis_min = 0
if x1 != x_axis_min:
    # 두 점 사이의 직선 방정식 구하기
    y_at_x_min = (y2 - y1) / (x2 - x1) * (x_axis_min - x1) + y1
    x1, y1 = x_axis_min, int(y_at_x_min)


# 선 인식 문제로 이미지 중간에서 끊길 경우 끝까지 그리기 위함
x_axis_max = img.shape[1] - 1
if x2 != x_axis_max:
    # 두 점 사이의 직선 방정식 구하기
    y_at_x_max = (y2 - y1) / (x2 - x1) * (x_axis_max - x1) + y1
    x2, y2 = x_axis_max, int(y_at_x_max)

print('parallel_line:', (x1, y1), (x2, y2))
parallel_line_y = (y1 + y2) // 2
print('parallel_line_y:', parallel_line_y)
print(f'parallel_equation: y={parallel_line_y}')
cv.line(original_img, (x1, y1), (x2, y2), (255, 255, 0), 2)

# 접점 계산
"""
(변수 참고)
circle_top: (x3, -y3)
circle_left: (x2, -y1)
circle_right: (x1, -y1)
circle_bottom: (x3, -y2)

타원 방정식은 다음과 같음
(x - x3) ** 2 / (x1 - x3) ** 2 + (y + y1) ** 2 / (-y3 + y1) ** 2 = 1

x에 대한 식으로 정리하면
_tmp_sqrt_in = (1 - ((y + y1) ** 2) / (-y3 + y1) ** 2) * (x1 - x3) ** 2
x = np.sqrt(_tmp_sqrt_in) + x3
"""

x1 = circle_right[0]
y1 = -circle_right[1]
x2 = circle_left[0]
y2 = -circle_bottom[1]
x3 = circle_top[0]
y3 = -circle_top[1]

y = parallel_line_y

_tmp_contact_x_sqrt_in = (1 - ((y + y1) ** 2) / (-y3 + y1) ** 2) * (x1 - x3) ** 2
contact_x = int(np.sqrt(_tmp_contact_x_sqrt_in) + x3)

contact_point = (contact_x, parallel_line_y)
print('contact_point:', contact_point)
cv.circle(img, contact_point, 3, (0, 0, 0), 0)

# @@@ 접점 이용해서 접선 구하기
"""
접점을 다음과 같이 약속함: (x4, -y4)

접선은 아래 방정식으로 표현 됨
(x4 - x3) * (x - x3) / (x1 - x3) ** 2 + (-y4 + y1) * (y + y1) / (-y3 + y1) ** 2 = 1

y에 대한 식으로 정리하면
left_eq = (1 - (x4 - x3) * (x - x3) / (x1 - x3) ** 2)
right_eq = (-y3 + y1) ** 2 / (-y4 + y1)
y = left_eq * right_eq - y1
"""

x4 = contact_point[0]
y4 = -contact_point[1]

contact_x1 = contact_x - 50  # 임의의 점을 설정, 접선으로부터 50픽셀 이전 위치
left_eq = (1 - (x4 - x3) * (contact_x1 - x3) / (x1 - x3) ** 2)
right_eq = (-y3 + y1) ** 2 / (-y4 + y1)
contact_y1 = int(left_eq * right_eq - y1)

contact_x2 = contact_x + 50  # 임의의 점을 설정, 접선으로부터 50픽셀 이후 위치
left_eq = (1 - (x4 - x3) * (contact_x2 - x3) / (x1 - x3) ** 2)
right_eq = (-y3 + y1) ** 2 / (-y4 + y1)
contact_y2 = int(left_eq * right_eq - y1)

contact_line = ((contact_x1, contact_y1), (contact_x2, contact_y2))
print('contact line point:', contact_line)
cv.line(original_img, contact_line[0], contact_line[1], (100, 0, 200), 2)

# @@@ 구해진 값을 이용해서 각도 계산
x1, y1, x2, y2 = parallel_line
(x3, y3), (x4, y4) = contact_line

# get angle
numerator = (x1 - x2) * (x3 - x4) + (y1 - y2) * (y3 - y4)
denominator = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) * np.sqrt((x3 - x4) ** 2 + (y3 - y4) ** 2)
angle = np.arccos(numerator / denominator)
angle_degree = np.rad2deg(angle)
angle_degree_left = 180 - angle_degree

print('angle:', angle_degree_left)

img_pil = Image.fromarray(img)
draw = ImageDraw.Draw(img_pil)

fontpath = './d2coding.ttc'
font = ImageFont.truetype(fontpath, 18)
draw.text((10, 10), f'θ={angle_degree_left:.2f}', font=font, fill=(255, 0, 255))
img = np.array(img_pil)


cv.imshow("result", img)
cv.waitKey(0)