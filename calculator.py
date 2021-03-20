import numpy as np

def get_elipse_axis(circle_img):
    """Get major, minor axis of elipse image.
    Background color of image must be black (rgb = 0, 0, 0) and this function find start point of non-black pixels

    Args:
        circle_img (numpy.array): Numpy array image
    """
    shape_y, shape_x, _ = circle_img.shape

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

    return circle_top, circle_right, circle_bottom, circle_left

def extend_parallel_line(img, parallel_line):
    """extend parallel line point when detected line is shorter than width of image

    Args:
        parallel_line (tuple<int, int, int, int>): parallel line start, end point (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = parallel_line

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

    return x1, y1, x2, y2

def get_contact_point(elipse_axies, parallel_line):
    """[summary]  TODO:

    Args:
        elipse_axies ([type]): [description]
        parallel_line ([type]): [description]
    """

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
    circle_top, circle_right, circle_bottom, circle_left = elipse_axies
    x1 = circle_right[0]
    y1 = -circle_right[1]
    x2 = circle_left[0]
    y2 = -circle_bottom[1]
    x3 = circle_top[0]
    y3 = -circle_top[1]

    y = (parallel_line[1] + parallel_line[3]) // 2  # parallel_line_y

    _tmp_contact_x_sqrt_in = (1 - ((y + y1) ** 2) / (-y3 + y1) ** 2) * (x1 - x3) ** 2
    contact_x = int(np.sqrt(_tmp_contact_x_sqrt_in) + x3)

    contact_point = (contact_x, y)
    return contact_point

def get_contact_line(contact_point, elipse_axies):
    """[summary]

    Args:
        contact_point ([type]): [description]
        elipse_axies ([type]): [description]
    """
    circle_top, circle_right, circle_bottom, circle_left = elipse_axies
    """
    접점을 다음과 같이 약속함: (x4, -y4)

    접선은 아래 방정식으로 표현 됨
    (x4 - x3) * (x - x3) / (x1 - x3) ** 2 + (-y4 + y1) * (y + y1) / (-y3 + y1) ** 2 = 1

    y에 대한 식으로 정리하면
    left_eq = (1 - (x4 - x3) * (x - x3) / (x1 - x3) ** 2)
    right_eq = (-y3 + y1) ** 2 / (-y4 + y1)
    y = left_eq * right_eq - y1
    """
    contact_x = contact_point[0]
    x1 = circle_right[0]
    y1 = -circle_right[1]
    x2 = circle_left[0]
    y2 = -circle_bottom[1]
    x3 = circle_top[0]
    y3 = -circle_top[1]

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

    return contact_line

def get_contact_angle(parallel_line, contact_line):
    """[summary]

    Args:
        parallel_line ([type]): [description]
        contact_line ([type]): [description]
    """
    x1, y1, x2, y2 = parallel_line
    (x3, y3), (x4, y4) = contact_line

    # get angle
    numerator = (x1 - x2) * (x3 - x4) + (y1 - y2) * (y3 - y4)
    denominator = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) * np.sqrt((x3 - x4) ** 2 + (y3 - y4) ** 2)
    angle = np.arccos(numerator / denominator)
    angle_degree_right = np.rad2deg(angle)
    angle_degree = 180 - angle_degree_right
    return angle_degree
