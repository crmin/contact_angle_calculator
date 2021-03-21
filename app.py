# import io

import cv2
from flask import Flask, render_template, request

from processor import *
from calculator import *


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST':
        template_variables = {}

        form_image = request.files.get('image')
        # image_binary = io.ByteIO(image.read())
        try:
            img = load_img_from_binary(form_image.read())
        except cv2.error:
            return render_template(
                'result.html',
                error=True,
                error_msg=(
                    'Occurred error when process image.<br>'
                    'Please check file is correct image.'
                )
            )
        form_image.seek(0)
        original_img = load_img_from_binary(form_image.read())

        circle_img = find_circle(img)
        elipse_axies = get_elipse_axis(circle_img)
        circle_top, circle_right, circle_bottom, circle_left = elipse_axies

        if request.form.get('show_ellipse_axies') is not None:
            draw_elipse_axises(img, elipse_axies)

        template_variables.update({
            'circle': {
                'top': circle_top,
                'bottom': circle_bottom,
                'left': circle_left,
                'right': circle_right,
            },
        })


        # @@@ 파란 원과 수평선의 접점 계산
        # 수평선은 이론상 x축에 평행해야하지만 실제로는 다를 수 있으므로 왼쪽 끝과 오른쪽 끝 점의 y좌표를 구해서 평균으로 사용

        line_img = find_parallel_line(img)

        parallel_line = detect_parallel_line(line_img)
        x1, y1, x2, y2 = parallel_line

        # 선 인식 문제로 중간부터 그릴 경우 처음부터 그리기 위함
        parallel_line = extend_parallel_line(img, parallel_line)
        parallel_line_y = (parallel_line[1] + parallel_line[3]) // 2
        template_variables.update({
            'parallel': {
                'line': (parallel_line[:2], parallel_line[2:]),
                'line_y': parallel_line_y,
                'parallel_equation': f'y={parallel_line_y}'
            },
        })

        if request.form.get('show_detected_line') is not None:
            draw_detected_parallel_line(img, parallel_line)

        # 접점 계산
        contact_point = get_contact_point(elipse_axies, parallel_line)
        template_variables['contact_point'] = contact_point
        if request.form.get('show_contact_point') is not None:
            draw_contact_point(img, contact_point)

        # @@@ 접점 이용해서 접선 구하기
        contact_line = get_contact_line(contact_point, elipse_axies)
        template_variables['contact_line_point'] = contact_line
        if request.form.get('show_contact_line') is not None:
            draw_contact_line(img, contact_line)

        # @@@ 구해진 값을 이용해서 각도 계산
        angle_degree = get_contact_angle(parallel_line, contact_line)
        template_variables['angle'] = angle_degree

        if request.form.get('show_contact_angle') is not None:
            img = draw_angle(img, angle_degree)
        return render_template(
            'result.html',
            original_image=convert_image_to_base64(original_img),
            result_image=convert_image_to_base64(img),
            **template_variables,
        )

if __name__ == '__main__':
    # If you want to change port number, change below code
    app.run(host='0.0.0.0', port='51273')
