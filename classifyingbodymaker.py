import matplotlib
matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt
plt.ioff()

import numpy as np
import cv2
import math
from skimage import measure
import matplotlib.patches as mpatches
from skimage import (io, color, filters, measure, morphology, segmentation, util)

class EllipseResult:
    def __init__(self, props, eccentricity):
        self.props = props
        self.eccentricity = eccentricity

def doRegionProps(bodymarker_img):
    labeled_image = measure.label(bodymarker_img)
    segmented_raw = morphology.remove_small_objects(labeled_image, 24)
    regions = measure.regionprops(segmented_raw)

    nipple_result = bodymarker_img.copy()

    # 그림그리기
    fig, ax = plt.subplots()
    ax.imshow(nipple_result, cmap=plt.cm.gray)

    height, width = bodymarker_img.shape[:2]
    half_width = width/2
    max_width = int(half_width + width*0.1)
    min_width = int(half_width - width*0.1)
    
    arr = []
    
    for props in regions:
        centroid = props.centroid
        x0 = centroid[1]
        minor_length = props.axis_minor_length
        major_length = props.axis_major_length
        eccentricity = math.sqrt(1-((minor_length/major_length)**2))
        if min_width <= x0 <= max_width:
            arr.append(EllipseResult(props, eccentricity))
    
    if arr is None:
        print('failed')
        return 'failed'

    if len(arr) >= 2:
        sorted_arr = sorted(arr, key=lambda x: (x.eccentricity, x.props.area_bbox))
        ref_obj = sorted_arr[0].props
    else:
        ref_obj = arr[0].props

    minr, minc, maxr, maxc = ref_obj.bbox
    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(rect)

    plt.show()

    reference_point = ref_obj
    return reference_point