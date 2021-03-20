import sys

import cv2 as cv
import numpy as np
from PIL import ImageFont, ImageDraw, Image

from processor import *
from calculator import *


if __name__ == '__main__':
    img = load_img(sys.argv[1])

    circle_img = find_circle(img)
    elipse_axies = get_elipse_axis(circle_img)
    circle_top, circle_right, circle_bottom, circle_left = elipse_axies

    draw_elipse_axises(img, elipse_axies)

    print('circle_top:', circle_top)
    print('circle_bottom:', circle_bottom)
    print('circle_left:', circle_left)
    print('circle_right:', circle_left)


    # @@@ 파란 원과 수평선의 접점 계산
    # 수평선은 이론상 x축에 평행해야하지만 실제로는 다를 수 있으므로 왼쪽 끝과 오른쪽 끝 점의 y좌표를 구해서 평균으로 사용

    line_img = find_parallel_line(img)

    parallel_line = detect_parallel_line(line_img)
    x1, y1, x2, y2 = parallel_line

    # 선 인식 문제로 중간부터 그릴 경우 처음부터 그리기 위함
    parallel_line = extend_parallel_line(img, parallel_line)
    parallel_line_y = (parallel_line[1] + parallel_line[3]) // 2

    print('parallel_line:', parallel_line[:2], parallel_line[2:])
    print('parallel_line_y:', parallel_line_y)
    print(f'parallel_equation: y={parallel_line_y}')

    draw_detected_parallel_line(img, parallel_line)

    # 접점 계산
    contact_point = get_contact_point(elipse_axies, parallel_line)
    print('contact_point:', contact_point)
    draw_contact_point(img, contact_point)

    # @@@ 접점 이용해서 접선 구하기
    contact_line = get_contact_line(contact_point, elipse_axies)
    print('contact line point:', contact_line)
    draw_contact_line(img, contact_line)

    # @@@ 구해진 값을 이용해서 각도 계산
    angle_degree = get_contact_angle(parallel_line, contact_line)
    print('angle:', angle_degree)

    img = draw_angle(img, angle_degree)

    cv.imshow("result", img)
    cv.waitKey(0)
