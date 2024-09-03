import cv2


def preds_vis(img, preds, color=(0, 0, 255), only_keep_cat=False, keep_cat=[]):
    '''
    img: image data
    preds: Yolo preds
    color: box and cat name vis color
    only_keep_cat: whether to vis the cat only in keep_cat
    keep_cat: the list of cats want to vis, only enable when only_keep_cat is True
    '''
    bboxes = preds.boxes
    names = preds.names
    for bbox in bboxes:
        cat_name = names[int(bbox.cls)]
        if only_keep_cat:
            if cat_name in keep_cat:
                x1, y1, x2, y2 = bbox.xyxy[0]
                sp, ep = (int(x1), int(y1)), (int(x2), int(y2))
                cv2.rectangle(img, sp, ep, color=color, thickness=1)
                cv2.putText(img, cat_name, sp, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
        else:
            x1, y1, x2, y2 = bbox.xyxy[0]
            sp, ep = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(img, sp, ep, color=color, thickness=1)
            cv2.putText(img, cat_name, sp, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
    return img


def gts_vis(img, annos, color=(0, 255, 0), only_keep_cat=False, keep_cat=[]):
    '''
    img: image data
    annos: coco json format annos
    color: box and cat name vis color
    only_keep_cat: whether to vis the cat only in keep_cat
    keep_cat: the list of cats want to vis, only enable when only_keep_cat is True
    '''
    for anno_info in annos:
        cat_name = anno_info['category_name']
        if only_keep_cat:
            if cat_name in keep_cat:
                bbox = anno_info['bbox']
                x1, y1, w, h = bbox
                x2, y2 = x1 + w, y1 + h
                sp, ep = (x1, y1), (x2, y2)
                cv2.rectangle(img, sp, ep, color=color, thickness=1)
                cv2.putText(img, cat_name, sp, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
        else:
            bbox = anno_info['bbox']
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
            sp, ep = (x1, y1), (x2, y2)
            cv2.rectangle(img, sp, ep, color=color, thickness=1)
            cv2.putText(img, cat_name, sp, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
    return img


def bbox_vis(img, bbox, color=(0, 255, 0), cat_name='', cat_score=None):
    '''
    img: image data
    bbox: tuple of list of (x1, y1, x2, y2)
    color: box and cat name vis color
    cat_name: cat name
    '''
    x1, y1, x2, y2 = bbox
    sp, ep = (x1, y1), (x2, y2)
    cv2.rectangle(img, sp, ep, color=color, thickness=1)
    if cat_score: cat_name = cat_name + '{:.3f}'.format(cat_score)
    cv2.putText(img, cat_name, sp, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
    return img