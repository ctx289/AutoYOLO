import json
from pathlib import Path
from collections import defaultdict

def box_iou(box1, box2):
    """
    Calculate IoU of two bounding boxes.

    Args:
        box1: tuple or list, (x1, y1, x2, y2)
        box2: tuple or list, (x1, y1, x2, y2)

    Returns:
        float, IoU of two bounding boxes
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    iou = intersection / union if union > 0 else 0
    return iou

def match_boxes(preds, gts, iou_threshold=0.0):
    """
    Match predicted boxes to ground truth boxes based on IoU.

    Args:
        preds: list of tuples or lists, predicted boxes, each box is (x1, y1, x2, y2, score)
        gts: list of tuples or lists, ground truth boxes, each box is (x1, y1, x2, y2)
        iou_threshold: float, IoU threshold for matching boxes

    Returns:
        list of tuples, matched boxes, each box is (pred_box, gt_box)
    """
    matched_boxes = []
    pred_indices = list(range(len(preds)))
    gt_indices = list(range(len(gts)))

    # Sort predicted boxes by confidence score in descending order
    pred_indices.sort(key=lambda i: preds[i][4], reverse=True)

    # Match predicted boxes to ground truth boxes
    for i in pred_indices:
        pred_box = preds[i][:4]
        best_iou = 0
        best_j = -1

        for j in gt_indices:
            gt_box = gts[j]
            iou = box_iou(pred_box, gt_box)

            if iou > iou_threshold and iou > best_iou:
                best_iou = iou
                best_j = j

        if best_j >= 0:
            matched_boxes.append((preds[i], gts[best_j]))
            gt_indices.remove(best_j)

    return matched_boxes

    
def img_level_metrics(coco_json, prediction_json, thresh_dict, iou_thresh=0.0):
    '''
    coco_json: coco json of gt
    prediction_json: coco api format output json
    thresh_dict: key is cat name, value is the given thresh. It only
                evaluates the cat given in thresh_dict, other cats in
                prediction_json will be considered as ok
    iou_thresh: iou threshold used to determine whether recall the gt box
    '''
    coco_json = Path(coco_json)
    prediction_json = Path(prediction_json)

    with coco_json.open('r') as f:
        gts_data = json.load(f)
        images = gts_data['images']
        annotations = gts_data['annotations']
        categories = gts_data['categories']
    
    imgid2annos = defaultdict(list)
    for anno_info in annotations:
        img_id = anno_info['image_id']
        imgid2annos[img_id].append(anno_info)

    catid2catname = {(cat_info['id']):cat_info['name'] for cat_info in categories if cat_info['name'] in thresh_dict}

    with prediction_json.open('r') as f:
        preds_data = json.load(f)
    imgid2preds = defaultdict(list)
    for pred_info in preds_data:
        img_id = pred_info['image_id']
        imgid2preds[img_id].append(pred_info)

    ok_img_num = 0
    ng_img_num = 0
    recall_img_num = 0
    real_recall_img_num = 0
    lucky_recall_img_num = 0
    miss_img_num = 0
    overkill_img_num = 0
    total_img_num = 0

    lucky_recall_infos = []
    real_recall_infos = []
    miss_img_infos = []
    overkill_img_infos = []

    for img_info in images:
        total_img_num += 1
        img_id = img_info['id']
        annos = imgid2annos[img_id]
        preds = imgid2preds[img_id]

        gt_bboxes = []
        pred_bboxes = []

        if img_info['sample_cat'] != 'OK':
            ng_img_num += 1

            for anno_info in annos:
                cat_name = anno_info['category_name']
                if cat_name in thresh_dict:
                    x, y, w, h = anno_info['bbox']
                    gt_bboxes.append([x, y, x+w, y+h, cat_name])
                
            for pred_info in preds:
                pred_cat = catid2catname[pred_info['category_id']]
                if pred_cat in thresh_dict:
                    thresh_ = thresh_dict[pred_cat]
                    if pred_info['score'] > thresh_:
                        x, y, w, h = pred_info['bbox']
                        score = pred_info['score']
                        pred_bboxes.append([x, y, x+w, y+h, score, pred_cat])

            if len(pred_bboxes) == 0:
                miss_img_num += 1
                miss_img_infos.append({'img_info': img_info, 'gt_bboxes': gt_bboxes})
                continue
            else:
                recall_img_num += 1
        
            matched_boxes = match_boxes(pred_bboxes, gt_bboxes, iou_threshold=iou_thresh)
            if len(matched_boxes) == 0:
                lucky_recall_img_num += 1
                lucky_recall_infos.append({'img_info': img_info, 'gt_bboxes': gt_bboxes, 'pred_bboxes': pred_bboxes})
            else:
                real_recall_img_num += 1
                real_recall_infos.append({'img_info': img_info, 'gt_bboxes': gt_bboxes, 'pred_bboxes': pred_bboxes})

        else:
            ok_img_num += 1
            for pred_info in preds:
                pred_cat = catid2catname[pred_info['category_id']]
                if pred_cat in thresh_dict:
                    thresh_ = thresh_dict[pred_cat]
                    if pred_info['score'] > thresh_:
                        x, y, w, h = pred_info['bbox']
                        score = pred_info['score']
                        pred_bboxes.append([x, y, x+w, y+h, score, pred_cat])
            
            if len(pred_bboxes) > 0:
                overkill_img_num += 1
                overkill_img_infos.append({'img_info': img_info, 'pred_bboxes': pred_bboxes})
    

    print(f'total_img_num: {total_img_num}, ok_img_num: {ok_img_num}, ng_img_num: {ng_img_num}')
    print(f'recall_img_num: {recall_img_num}, real_recall_img_num: {real_recall_img_num}, lucky_recall_img_num: {lucky_recall_img_num}')
    print(f'miss_img_num: {miss_img_num}')
    print(f'overkill_img_num: {overkill_img_num}')
    if ng_img_num != 0:
        print(f'escape rate: {miss_img_num/ng_img_num:.3}')
    else:
        print(f'escape rate: 0.0')
    if ng_img_num != 0:
        print(f'real escape rate: {(ng_img_num - real_recall_img_num)/ng_img_num:.3}')
    else:
        print(f'real escape rate: 0.0')
    if ok_img_num!=0:
        print(f'overkill rate: {overkill_img_num/ok_img_num:.3}')
    else:
        print(f'overkill rate: 0.0')

    return lucky_recall_infos, real_recall_infos, miss_img_infos, overkill_img_infos



if __name__ == "__main__":
    coco_json = Path('/youtu/xlab-team4/gleelin/Project/ultralytics/self_materials/bottle_exp/bottle_cap/dataset/val.json')
    prediction_json = Path('/youtu/xlab-team4/gleelin/Project/ultralytics/my_scripts/predictions.json')

    lucky_recall_infos, real_recall_infos, miss_img_infos, overkill_img_infos = img_level_metrics(coco_json, prediction_json, thresh_dict={'ZW': 0.0})