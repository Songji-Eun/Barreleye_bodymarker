import pydicom
import matplotlib
matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
from skimage import measure, morphology
import matplotlib.patches as mpatches
import pandas as pd
from scipy import ndimage
from extractingbodymarker import DistanceResult, distanceCalculator, doFeatureMatching, getCropCoords, isBodymarker
from classifyingbodymaker import EllipseResult, doRegionProps
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os



def crop_and_resize(image, min_x, min_y, max_x, max_y, scale_factor=1.0):
    cropped_image = image[min_y:max_y, min_x :max_x]

    if scale_factor != 1.0:
        interpolation = cv2.INTER_LINEAR  # Interpolation method (can be changed)
        tmp = cropped_image.shape
        h = tmp[0]
        w = tmp[1]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        cropped_image = cv2.resize(cropped_image, (new_w, new_h), interpolation=interpolation)


    return cropped_image


def crop_with_magnify(image, x1, y1, x2, y2):
    # 확대할 영역을 선택합니다. 예: (x 시작, y 시작, 너비, 높이)
    # 이 값은 실제 필요에 따라 조정해야 합니다.
    wh = max((x2-x1), (y2-y1))
    x, y, w, h = x1, y1, wh, wh  # 예시 좌표 및 크기

    cropped = image[y:y+h, x:x+w]
    fig, ax = plt.subplots()
    ax.imshow(image)

    # 확대된 영역을 표시합니다.
    axins = ax.inset_axes([x, y, w, h])
    axins.imshow(image[y:y + h, x:x + w], interpolation='none')
    axins.set_xlim(x, x + w)
    axins.set_ylim(y, y + h)

    ax.axis('off')
    axins.axis('off')

    # Figure를 이미지로 변환
    canvas = FigureCanvas(fig)
    buf = io.BytesIO()
    canvas.print_png(buf)
    buf.seek(0)

    # BytesIO에서 이미지로 읽기
    image = plt.imread(buf)

    plt.close(fig)  # Close the figure to free resources

    return image



def backgroundRemoval(image):
    original_image = image.copy()
    blurred = cv2.GaussianBlur(image, (11, 11), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    tmp = morphology.remove_small_holes(closed)

    thick_edges = np.where(tmp, original_image, 0) #원래이미지랑 OR시키는 것과 비슷한 거
    # 결과 시각화
    plt.imshow(thick_edges, cmap='gray')
    plt.title('Thick Lines1')
    plt.show()
    
    return thick_edges



def findEllipseFoci(image):
    img_for_line = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
    contours, hier = cv2.findContours(image.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if len(contour) >= 5:
            # draw circle at center
            ellipse = cv2.fitEllipse(contour)
            cv2.ellipse(img_for_line, ellipse, (0, 255, 0), 1)
            (xc,yc),(d1,d2),angle = ellipse
            cv2.circle(img_for_line, (int(xc),int(yc)), 1, (0, 0, 255), -1)

                # draw major axis line
            rmajor = max(d1, d2) / 2
            angle_rad = math.radians(angle + 90)
            x1 = int(xc - rmajor * math.cos(angle_rad))
            y1 = int(yc - rmajor * math.sin(angle_rad))
            x2 = int(xc + rmajor * math.cos(angle_rad))
            y2 = int(yc + rmajor * math.sin(angle_rad))
            cv2.line(img_for_line, (x1, y1), (x2, y2), (0, 0, 255), 3)

            # draw minor axis line
            rminor = min(d1, d2) / 2
            angle_rad_minor = math.radians(angle)
            x1_minor = int(xc - rminor * math.cos(angle_rad_minor))
            y1_minor = int(yc - rminor * math.sin(angle_rad_minor))
            x2_minor = int(xc + rminor * math.cos(angle_rad_minor))
            y2_minor = int(yc + rminor * math.sin(angle_rad_minor))
            cv2.line(img_for_line, (x1_minor, y1_minor), (x2_minor, y2_minor), (255, 0, 0), 3)

            # draw foci along major axis
            focus_distance = np.sqrt(abs(rmajor**2 - rminor**2))
            focus1_x = int(xc + math.cos(angle_rad) * focus_distance)
            focus1_y = int(yc + math.sin(angle_rad) * focus_distance)
            focus2_x = int(xc + math.cos(angle_rad + math.pi) * focus_distance)
            focus2_y = int(yc + math.sin(angle_rad + math.pi) * focus_distance)
            cv2.circle(img_for_line, (focus1_x, focus1_y), 2, (255, 255, 255), -1)
            cv2.circle(img_for_line, (focus2_x, focus2_y), 2, (255, 255, 255), -1)

            
            #result = cv2.addWeighted(back_img, 0.8, img_for_line, 1, 0)

    plt.imshow(img_for_line)
    plt.title('major minor')
    plt.axis('off')  # 축 표시를 끕니다.
    plt.show()

    return





def HoughLine(img):
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    threshold_value = 128  # 일반적으로 128을 사용하지만, 적절한 값을 실험적으로 찾아야 할 수 있습니다.
    _, binary_image = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)

    kernel_size = 5
    blur_gray = cv2.GaussianBlur(binary_image,(kernel_size, kernel_size),0)

    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 30  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 9  # minimum number of pixels making up a line
    max_line_gap = 2  # maximum gap in pixels between connectable line segments

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
    return lines




def findLines(image):
    houghimg = image.copy()
    color_img = cv2.cvtColor(houghimg, cv2.COLOR_GRAY2BGR)
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    
    threshold_value = 128 
    _, binary_image = cv2.threshold(gray_img, threshold_value, 255, cv2.THRESH_BINARY)

    kernel_size = 5
    blur_gray = cv2.GaussianBlur(binary_image, (kernel_size, kernel_size), 0)
    
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    rho = 1
    theta = np.pi/180
    threshold = 30
    min_line_length = 10
    max_line_gap = 5
    line_image = np.copy(color_img) * 0 
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    print('line', lines)
    # 검출된 선을 원본 이미지에 그리기
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1,y1), (x2,y2), (255,255,255), 2)

    # 결과 이미지 보기
    lines_edges = cv2.addWeighted(color_img, 0.8, line_image, 1, 0)

    plt.figure(figsize=(8,8))
    plt.imshow(cv2.cvtColor(lines_edges, cv2.COLOR_BGR2RGB))
    plt.title('Detected Lines')
    plt.axis('off')  # 축 표시를 끕니다.
    plt.show()


class barreleye_bodymarker_visbus():
    def __init__(self):
        self.position_list = ['Right Lymph node', 'Right nipple', 'Right UIQ', 'Right UOQ', 'Right LIQ', 'Right LOQ'
            , 'Left Lymph node', 'Left nipple', 'Left UIQ', 'Left UOQ', 'Left LIQ', 'Left LOQ']

    def extract_bodymarker(self, dcm_image):
        bodymarker_img = None

        query_folder_path = './Query_Files/PNGFiles'
        query_image_files = [f for f in os.listdir(query_folder_path) if f.endswith('.png')]
        
        query_distCalObj = []
        for query_file_name in query_image_files:
            file_path = os.path.join(query_folder_path, query_file_name)
            
            query_image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2GRAY)
            RL_label = query_file_name.split('_')[-1].split('.')[0]

            query_obj_tmp = distanceCalculator(query_image, RL_label, dcm_image)
            query_distCalObj.append(query_obj_tmp)
            print(query_obj_tmp.avg_distance)  #debugging
        

        query_min_distObj = (sorted(query_distCalObj, key = lambda x:x.avg_distance))[0] # Sort them in the order of their distance, 제일 작은 것 가져오기

        #############만약 최소 distance가 1500을 넘으면 bodymarker가 없다고 판단#####################
        check_marker = isBodymarker(query_min_distObj, threshold=1500)
        if check_marker is False:
            return None, 'No image'  #bodymarker, RorL 순서로 return
        ####################################################################################
        
        dst_transformed = doFeatureMatching(query_min_distObj, dcm_image)

        # image crop
        min_y, max_y, min_x, max_x = getCropCoords(dst_transformed)    #output : array[min_y, max_y, min_x, max_x]
        
        #crop and magnify를 하니까 축 모양까지 포함되어서 resizing하는 거로 해봤어여
        #scale_factor = 150/w
        #bodymarker_img = crop_and_resize(dcm_image, min_x, min_y, max_x, max_y, scale_factor)
        
        bodymarker_img = dcm_image[min_y:max_y, min_x:max_x]
        
        return bodymarker_img, query_min_distObj.RorL



    def classify_bodymarker(self, bodymarker_img):
        lesion_position = None
        ######################reference point 찾기 - region props############################
        reference_point = doRegionProps(bodymarker_img)
        point_y, point_x = reference_point.centroid
        ####################################################################################


        ##################################line detection####################################
        findLines(bodymarker_img)
        ####################################################################################


        ###########################ellipse region으로 유방 위치 찾기############################
        findEllipseFoci(bodymarker_img)
        ####################################################################################
        

        
        if bodymarker_img is None:
            print('Failed')
            return 'Failed'
        
        return lesion_position
