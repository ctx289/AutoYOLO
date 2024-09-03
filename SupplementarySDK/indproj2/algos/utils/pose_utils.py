import logging
import os
import re


def get_pose(image_path, product_name=""):
    """ get pose num for image path or image name

    Args:
        image_path (str): image_path

    Returns:
        int: pose
    """
    image_name = os.path.basename(image_path)
    # if product_name == "DEYI":
    #     pose = image_name.split("_")[0]
    # else:
    pose = re.search(r"P(\d+)", image_name).group(1)
    return int(pose)


def get_group(image_path, product_name=""):
    """ get group/cameraid num for image path or image name

    Args:
        image_path (str): image_path

    Returns:
        int: camera id / group
    """
    image_name = os.path.basename(image_path)
    # if product_name == "DEYI":
    #     group = re.search(r"Cam(\d+)", image_name).group(1)
    # else:
    if product_name == "FUCHI":
        group = re.search(r"G(\d+)", image_name).group(1)
    else:
        group = re.search(r"C(\d+)", image_name).group(1)
    return int(group)


def get_product(image_path, strict=False, **kwargs):
    """get product string
        according to name rule, product starts with PI endswiths _
        such as, PI139_ PIC_ PID_

    Args:
        image_path (str): image_path

    Returns:
        product string
    """
    product_name = ""
    for surffix in ["_", "-"]: # "\." if compatible with .jpg
        cur_product_name = re.search(r'PI(.*?){}'.format(surffix), image_path)
        if cur_product_name is None:
            continue
        cur_product_name = cur_product_name.group(1)
        if not product_name or len(cur_product_name) < len(product_name):
            product_name = cur_product_name

    if not product_name:
        if strict:
            raise ValueError(
                f"Cannot find product name (_PIxxx_ or _PIxxx-) for image {image_path}"
            )
    return product_name


def get_product_pose(image_path, strict=True, **kwargs):
    """ get_product_pose
    return

    Args:
        image_path (str): path of image
        strict (bool, optional): whether report error
            when product name not found. Defaults to True.

    Raises:
        AttributeError: if not PIxxx, product name found
        AttributeError: if not Pxxx, pose name found

    Returns:
        str: product_pose name
        int: not product_name, only pose
    """
    prod_name = get_product(image_path, strict=strict, **kwargs)

    try:
        pose_name = get_pose(image_path, **kwargs)
    except Exception as e:
        logging.error(f"Cannot get pose (Pxxx) for image {image_path}")
        raise AttributeError(e)

    if prod_name is None:
        return int(pose_name)

    return f"{prod_name}_{pose_name}"


def get_color(image_path, product_name="", strict=False):
    """ get color for image path or image name
        according to data name rule, the color is [Y+int]
        such as Y1, Y2, Y3...

    Args:
        image_path (str): image_path

    Returns:
        int: color
    """
    image_name = os.path.basename(image_path)
    color = -1
    try:
        color = re.search(r"Y(\d+)", image_name).group(1)
    except Exception as e:
        logging.warning(f"Cannot get color (Y+int) for image {image_path}")
        if strict:
            raise AttributeError(e)

    return int(color)
