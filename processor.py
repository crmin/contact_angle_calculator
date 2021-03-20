import sys

import cv2 as cv
import numpy as np
from PIL import ImageFont, ImageDraw, Image


def load_img(path):
    img = cv.imread(path)
    return img

def load_img_from_binary(binary):
    encoded_img = np.fromstring(binary, dtype=np.uint8)
    img = cv.imdecode(encoded_img, cv.IMREAD_COLOR)
    return img

def find_circle(img):
    hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)

    blue_lower = np.array([94, 80, 2], np.uint8)
    blue_upper = np.array([120, 255, 255], np.uint8)
    blue_mask = cv.inRange(hsv, blue_lower, blue_upper)
    circle_img = cv.bitwise_and(img, img, mask=blue_mask)

    return circle_img

def draw_elipse_axises(img, elipse_axies):
    circle_top, circle_right, circle_bottom, circle_left = elipse_axies
    cv.circle(img, circle_top, 3, (0, 0, 255), -1)
    cv.circle(img, circle_bottom, 3, (0, 255, 0), -1)
    cv.circle(img, circle_left, 3, (255, 0, 0), -1)
    cv.circle(img, circle_right, 3, (0, 255, 255), -1)

def find_parallel_line(img):
    hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)

    red_lower = np.array([-10, 100, 100])
    red_upper = np.array([10, 255, 255])
    red_mask = cv.inRange(hsv, red_lower, red_upper)
    line_img = cv.bitwise_and(img, img, mask=red_mask)

    return line_img

def detect_parallel_line(line_img):
    gray = cv.cvtColor(line_img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    parallel_line = None

    for idx, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]
        parallel_line = line[0]
        break

    return parallel_line

def draw_detected_parallel_line(original_img, parallel_line):
    cv.line(original_img, parallel_line[:2], parallel_line[2:], (255, 255, 0), 2)

def draw_contact_point(original_img, contact_point):
    cv.circle(original_img, contact_point, 3, (0, 0, 0), 0)

def draw_contact_line(original_img, contact_line):
    cv.line(original_img, contact_line[0], contact_line[1], (100, 0, 200), 2)

def draw_angle(img, angle, fontpath='./d2coding.ttc'):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)

    font = ImageFont.truetype(fontpath, 18)
    draw.text((10, 10), f'θ={angle:.2f}', font=font, fill=(255, 0, 255))
    img = np.array(img_pil)
    return img
