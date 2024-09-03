""" basic rule parts """
import copy
import logging
import os
import math
import time
import copy
import cv2

import numpy as np

from .rule_builder import RULE_PARTS
from .basic_rule_parts import BasicRules

@RULE_PARTS.register_module()
class FilterBySizeMeasurement(BasicRules):
    """ Filter box which detection class is not in ScoreThresh
        which means the box class is ok class
        This function is part of FilterByKeyAndScore, writing seperatly for
        funnel analysis (do this first then do FilterByKeyAndScore).
    """

    def __init__(self, thresh=None):
        self.kernel = np.ones((15, 15), np.uint8)
        self.colors = [(0, 255, 255), (255, 0, 255), (200, 255, 100), (255, 255, 0)]
        self.classname = 'SIZENG'
        self.thresh = thresh
        self.real_thresh = 0.53*1000/9.1
    
    def meature_distance_v2(self, img, enhance=True, debug=False):

        # ============== 前处理
        # RGB转灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 固定阈值分割
        ret, thresh1 = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        # 膨胀
        thresh2 = cv2.dilate(thresh1, self.kernel, iterations=2)
        # 膨胀后mask取反
        thresh3 = 255 - thresh2
        # 腐蚀
        thresh4 = cv2.erode(thresh2, self.kernel, iterations=2)

        # ============== enhance, 找到最大连通域
        if enhance:
            """
            1. 查找轮廓
            2. 将轮廓转换为polygon,只取最大的那个polygon
            3. 只取最大的那个polygon,将polygon转换为mask
            """
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh4, connectivity=8)
            areas = stats[:, 4]
            max_label = np.argmax(areas[1:]) + 1
            thresh4 = np.zeros_like(labels).astype(np.uint8)
            thresh4[labels == max_label] = 255
        
        # ============== 检测圆
        # gray边缘检测
        edges_2 = cv2.Canny(gray, 100, 200)
        # 边缘掩膜后处理
        thresh3[thresh3==255]=1
        edges_2 = thresh3 * edges_2
        # 霍夫变换圆检测
        circles = cv2.HoughCircles(edges_2, cv2.HOUGH_GRADIENT, 1, 100, param1=200, param2=10, minRadius=26, maxRadius=30)
        # 根据检测到圆的信息，画出每一个圆
        points = []
        try:
            for circle in circles[0]:
                # 坐标行列
                x = int(circle[0])
                y = int(circle[1])
                r = int(circle[2])
                # 记录圆心
                points.append((x, y))
        except:
            return {}
        # 过滤部份圆
        pass
        # 对圆排序，绘制圆心矩形
        if len(circles[0]) == 4:
            points = np.array(points)
            center = np.mean(points, axis=0).astype(int)
            angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
            sorted_idx = np.argsort(angles)
            sorted_points = points[sorted_idx]
        
        # ============== 计算未投影前平均长度 + 投影
        result={}
        if len(circles[0]) == 4:
            for i in range(4):
                if i == 0 or i == 2:
                    x_base = np.minimum(sorted_points[i][0], sorted_points[i+1][0])
                    x_range = np.abs(sorted_points[i][0] -  sorted_points[i+1][0])
                    cnt = []
                    for x_gap in np.round(np.linspace(0, x_range, 30)).astype(int)[6:26]:
                        x = x_base + x_gap
                        if i == 0:
                            # ATTENTION! the order of x y is (y, x).
                            cnt.append(((thresh4[:center[1],:][:,x]) / 255).sum())
                        elif i == 2:
                            cnt.append(((thresh4[center[1]:,:][:,x]) / 255).sum())    
                    # 排序, 去掉最大和最小10%, 剩下的取平均
                    cnt = sorted(cnt)
                    n = len(cnt)
                    remove_num = int(n * 0.1)
                    cnt_ = cnt[remove_num:n-remove_num]
                    avg = np.mean(cnt_)
                    # 夹角
                    angle = math.atan2(np.abs(sorted_points[i][1] - sorted_points[i+1][1]), np.abs(sorted_points[i][0] - sorted_points[i+1][0])) * 180
                    # 夹角的余弦值
                    cos_value = math.cos(angle / 180 * math.pi)
                    # 投影
                    cos_avg = avg * cos_value
                    if debug:
                        print('物料偏移的角度{} : {:.3f}'.format(i, angle))
                        print('投影前平均长度{} : {:.2f}'.format(i, avg))
                        print('投影后平均长度{} : {:.2f}'.format(i, cos_avg))
                elif i == 1 or i == 3:
                    y_base = np.minimum(sorted_points[i][1], sorted_points[(i+1)%4][1])
                    y_range = np.abs(sorted_points[i][1] -  sorted_points[(i+1)%4][1])
                    cnt = []
                    for y_gap in np.round(np.linspace(0, y_range, 30)).astype(int)[6:26]:
                        y = y_base + y_gap
                        if i == 1:
                            cnt.append(((thresh4[:,center[0]:][y,:]) / 255).sum())
                        elif i == 3:
                            cnt.append(((thresh4[:,:center[0]][y,:]) / 255).sum())
                    # 排序, 去掉最大和最小10%, 剩下的取平均
                    cnt = sorted(cnt)
                    n = len(cnt)
                    remove_num = int(n * 0.1)
                    cnt_ = cnt[remove_num:n-remove_num]
                    avg = np.mean(cnt_)
                    # 夹角
                    angle = math.atan2(np.abs(sorted_points[i][0] - sorted_points[(i+1)%4][0]), np.abs(sorted_points[i][1] - sorted_points[(i+1)%4][1])) * 180
                    # 夹角的余弦值
                    cos_value = math.cos(angle / 180 * math.pi)
                    # 投影
                    cos_avg = avg * cos_value
                    if debug:
                        print('物料偏移的角度{} : {:.3f}'.format(i, angle))
                        print('投影前平均长度{} : {:.2f}'.format(i, avg))
                        print('投影后平均长度{} : {:.2f}'.format(i, cos_avg))
                result[i] = cos_avg
        return result
    
    def generate_pred_list(self, length, filter, filter_by):
        pred_list=[]
        bbox = np.array([1, 1, 501, 501, 0.99])
        xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
        det_code = self.classname
        area = (ymax - ymin) * (xmax - xmin)
        boundary = None
        pred_list.append({
                "det_score": bbox[-1],
                "det_bbox": bbox[:-1].astype(np.int).tolist(),
                "det_code": det_code,
                "area": int(area),
                "length": int(length),
                "polygon": boundary,
                "filter": filter,
                "filter_by": filter_by,
            })
        return pred_list
    
    def feed_data(self, feed_dict, key, value):
        try:
            feed_dict[key].extend(value)
        except KeyError:
            feed_dict[key] = []
            feed_dict[key].extend(value)
        return feed_dict

    def __call__(self, feed_dict, key, common_info=None):
        result = self.meature_distance_v2(feed_dict['images'][0], enhance=True, debug=False)
        values = list(result.values())
        if (np.array(values) > self.thresh).any():
            # 检测为NG
            pred_list = self.generate_pred_list(max(values), filter=1, filter_by="high_ng")
            self.feed_data(feed_dict, 'preds', pred_list)
        else:
            if len(values)>0:
                # 检测为OK
                pred_list = self.generate_pred_list(max(values), filter=0, filter_by="low_det_ok")
                self.feed_data(feed_dict, 'preds', pred_list)
            else:
                # 检测失败
                pred_list = []
                self.feed_data(feed_dict, 'preds', pred_list)
        preds = feed_dict[key]
        return preds