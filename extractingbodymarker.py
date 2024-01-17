import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import cv2



class DistanceResult:
    def __init__(self, query_img, RorL, avg_distance, keypoints_query, keypoints_target, match_array):
        self.query_img = query_img
        self.RorL = RorL
        self.avg_distance = avg_distance
        self.keypoints_query = keypoints_query
        self.keypoints_target = keypoints_target
        self.match_array = match_array


def distanceCalculator(query_img, RorL, target_img):
    detector = cv2.SIFT.create(nfeatures=850, nOctaveLayers=11, contrastThreshold = 0.08)

    kp_query, desc_query = detector.detectAndCompute(query_img, None)
    kp_target, desc_target = detector.detectAndCompute(target_img, None)

    # ③ BFMatcher 생성, L1 거리, 상호 체크
    matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = matcher.match(desc_query, desc_target)
    matches = sorted(matches, key = lambda x:x.distance) # Sort them in the order of their distance
    best_n_matches_avg = matches[:40] # 그리고 10개 가져오기
    best_n_matches_cal = matches[:36]

    #distance 평균 계산
    distances = np.array([match.distance for match in best_n_matches_avg])
    average_distance = np.mean(distances)

    obj_calculated = DistanceResult(query_img, RorL, average_distance, kp_query, kp_target, best_n_matches_cal)
    return obj_calculated


def isBodymarker(query_min_distObj, threshold):
    if query_min_distObj.avg_distance > threshold:
        return False
    else:
        return True


def drawingMatchLine(dst_pt, w, mask, queryimg, kp_query, dcm_img, kp_target, match_arr):
    tmp_pt = dst_pt + (w, 0)
    matchesMask = mask.ravel().tolist()
    draw_params = dict(matchColor = (0,255,0), singlePointColor = None, matchesMask = matchesMask, flags = 2)
    res = cv2.drawMatches(queryimg,kp_query,dcm_img,kp_target,match_arr, None,**draw_params)
    res = cv2.polylines(res, [np.int32(tmp_pt)], True, (0,0,255),3, cv2.LINE_AA)
    return res


def doFeatureMatching(query_min_distObj, dcm_image):
    keypoints_query = query_min_distObj.keypoints_query
    keypoints_target = query_min_distObj.keypoints_target
    match_array = query_min_distObj.match_array
    query_img = query_min_distObj.query_img 
    h,w = query_img.shape 

    matched_query_pts = np.float32([ keypoints_query[m.queryIdx].pt for m in match_array ]).reshape(-1,1,2)
    matched_target_pts = np.float32([ keypoints_target[m.trainIdx].pt for m in match_array ]).reshape(-1,1,2)
    box_pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(matched_query_pts, matched_target_pts, cv2.RANSAC, 5.0)
    dst_transformed = cv2.perspectiveTransform(box_pts,M)

    ###########match line 그리고 박스치는거 (안쓰면 지워도됨)###############
    matched_result = drawingMatchLine(dst_transformed, w, mask, query_img, keypoints_query, dcm_image, keypoints_target, match_array)
    plt.imshow(matched_result)
    plt.show()
    #######################################################
    return dst_transformed


#Crop을 위해서 가로 세로의 max, min point를 반환하는 함수
def getCropCoords(pts_info):
    pts_tmp = np.int32(pts_info).reshape(-1,2)
    x_coords = pts_tmp[:,0]
    y_coords = pts_tmp[:,1]
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)
    return min_y, max_y, min_x, max_x