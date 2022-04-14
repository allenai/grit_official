import os
import cv2
import copy
import base64
import torch
import numpy as np
from io import BytesIO
from PIL import Image, ImageDraw
from typing import List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from dataclasses import dataclass, fields, field

import utils.rle as rle


def hex2rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2],16) for i in (0,2,4))


HEX_COLORS = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0',
    '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8',
    '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', 
    '#000000']*5
RGB_COLORS = [hex2rgb(h) for h in HEX_COLORS]


def read_image(img_path: str) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)


def save_image(img: np.ndarray, tgt_path: str) -> None:
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    cv2.imwrite(tgt_path,img)


def display_image(img: np.ndarray) -> None:
    img = Image.fromarray(img,'RGB')
    img.show()


def display_thumbnail(img: np.ndarray, max_size=(100,100)) -> None:
    img = Image.fromarray(img,'RGB')
    img.thumbnail(max_size)
    img.show()


def base64_encode(img: np.ndarray, max_size=(100,100)):
    img = Image.fromarray(img,'RGB')
    img.thumbnail(max_size)
    with BytesIO() as buffer:
        img.save(buffer, 'png')
        return base64.b64encode(buffer.getvalue()).decode()


def get_image_formatter(input_type):
    if input_type=='base64':
        def formatter(enc_img):
            return f'<img src="data:image/png;base64,{enc_img}">'
    else:
        raise NotImplementedError
    
    return formatter


def add_bboxes(img: np.ndarray, bboxes: List[list], color=(200,0,200), thickness=3) -> None:
    """
    bboxes: list of [x1,y1,x2,y2] coordinates
    """
    r,g,b = color
    for bbox in bboxes:
        x1,y1,x2,y2 = bbox
        cv2.rectangle(img,(x1,y1),(x2,y2),(r,g,b),thickness)


def add_seg(img: np.ndarray, masks: List[np.ndarray], colors=RGB_COLORS, alpha=0.2) -> None:
    """
    masks: list of binary masks with 0/1 values
    """
    for i,mask in enumerate(masks):
        img[mask==1,:] = (1-alpha)*img[mask==1,:] + alpha*np.array([colors[i]])


CONNECT_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (5, 11), (6, 12), (11, 13), (12, 14), (13, 15), (14, 16)
]
N_KEYPOINTS = 17
N_DIM = 3

# https://thadeusb.com/weblog/2010/10/10/python_scale_hex_color/
def clamp(val, minimum=0, maximum=255):
    if val < minimum:
        return minimum
    if val > maximum:
        return maximum
    return val

# https://thadeusb.com/weblog/2010/10/10/python_scale_hex_color/
def colorscale(hexstr, scalefactor):
    """
    Scales a hex string by ``scalefactor``. Returns scaled hex string.

    To darken the color, use a float value between 0 and 1.
    To brighten the color, use a float value greater than 1.

    >>> colorscale("#DF3C3C", .5)
    #6F1E1E
    >>> colorscale("#52D24F", 1.6)
    #83FF7E
    >>> colorscale("#4F75D2", 1)
    #4F75D2
    """

    hexstr = hexstr.strip('#')

    if scalefactor < 0 or len(hexstr) != 6:
        return hexstr

    r, g, b = int(hexstr[:2], 16), int(hexstr[2:4], 16), int(hexstr[4:], 16)

    r = int(clamp(r * scalefactor))
    g = int(clamp(g * scalefactor))
    b = int(clamp(b * scalefactor))

    return "#%02x%02x%02x" % (r, g, b)

# converted to numpy from https://pytorch.org/vision/main/_modules/torchvision/utils.html#draw_keypoints
def draw_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    connectivity: Optional[List[Tuple[int, int]]] = None,
    colors: Optional[Union[str, Tuple[int, int, int]]] = None,
    radius: int = 2,
    width: int = 3,
) -> np.ndarray:
    """
    Draws Keypoints on given RGB image.
    The values of the input image should be uint8 between 0 and 255.

    Args:
        image (nndarray): ndarray of shape (3, H, W) and dtype uint8.
        keypoints (ndarray): ndarray of shape (K, 2) the K keypoints location 
            in the format [x, y]
        connectivity (List[Tuple[int, int]]]): A List of tuple where,
            each tuple contains pair of keypoints to be connected.
        colors (str, Tuple): The color can be represented as
            PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``.
        radius (int): Integer denoting radius of keypoint.
        width (int): Integer denoting width of line connecting keypoints.

    Returns:
        img (ndarray[H, W, C]): Image ndarray of dtype uint8 with keypoints drawn.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"The image must be a ndarray, got {type(image)}")
    elif image.dtype != np.uint8:
        raise ValueError(f"The image dtype must be uint8, got {image.dtype}")
    elif len(image.shape) != 3:
        raise ValueError("Pass individual images, not batches")
    elif image.shape[2] != 3:
        raise ValueError("Pass an RGB image. Other Image formats are not supported")
    if keypoints.ndim != 2:
        raise ValueError("keypoints must be of shape (K, 2)")
    img_to_draw = Image.fromarray(image)
    draw = ImageDraw.Draw(img_to_draw)

    for kpt in keypoints:
        x1 = kpt[0] - radius
        x2 = kpt[0] + radius
        y1 = kpt[1] - radius
        y2 = kpt[1] + radius
        draw.ellipse([x1, y1, x2, y2], fill=colors, outline=None, width=0)

    if connectivity:
        for connection in connectivity:
            start_pt_x = keypoints[connection[0]][0]
            start_pt_y = keypoints[connection[0]][1]

            end_pt_x = keypoints[connection[1]][0]
            end_pt_y = keypoints[connection[1]][1]

            draw.line(
                ((start_pt_x, start_pt_y), (end_pt_x, end_pt_y)),
                width=width,
                fill=colorscale(colors, 2)
            )

    return np.array(img_to_draw)

def add_kp(img: np.ndarray, kps: List[list]) -> np.ndarray:
    for i,kp in enumerate(kps):
        k_array3d = np.reshape(np.array(kp),(N_KEYPOINTS,N_DIM))
        kp_xy = k_array3d[:,:2]
        k_vis = k_array3d[:,2]

        if k_vis.sum() == 0: # no keypoints are labeled for this person
            continue 

        visible_kps = np.argwhere(k_vis).ravel().tolist()
        connections = [c for c in CONNECT_SKELETON if set(c).issubset(set(visible_kps))]

        color = HEX_COLORS[i]
        img = draw_keypoints(img, kp_xy, connectivity=connections, colors=color, radius=5, width=4)

    return img


@dataclass
class GritIO:
    text: Optional[List[str]] = None
    boxes: Optional[List[List[int]]] = None
    segs: Optional[List[np.ndarray]] = None
    kps: Optional[List[List[float]]] = None
    sn: Optional[List[np.ndarray]] = None 


@dataclass
class GritVizParams:
    box_color: Tuple[int,int,int] = (0,0,200)
    box_thickness: int = 3
    seg_alpha: float = 0.3
    seg_colors: List[Tuple] = field(default_factory=lambda: RGB_COLORS)
    prefix: str = 'Text'


@dataclass
class GroundTruth(GritVizParams):
    box_color: Tuple[int,int,int] = (0,200,0)
    prefix: str = 'GT'
    

@dataclass
class Prediction(GritVizParams):
    box_color: Tuple[int,int,int] = (200,0,200)
    prefix: str = 'Pred'


@dataclass
class Input(GritVizParams):
    box_color: Tuple[int,int,int] = (0,0,200)
    prefix: str = 'Query'


@dataclass
class GritIOType:
    GT = GroundTruth()
    PRED = Prediction()
    INPUT = Input()


def grit_viz(
        img: np.ndarray,
        ios: GritIO,
        io_type: GritVizParams=GritIOType.PRED,
        text=None):
    img_ = copy.deepcopy(img)
    
    if text is None:
        text = dict()
    
    if ios.text is not None:
        text[io_type.prefix] = ios.text

    if ios.boxes is not None:
        add_bboxes(img_, ios.boxes, io_type.box_color, io_type.box_thickness)
    
    if ios.segs is not None:
        add_seg(img_, ios.segs, colors=io_type.seg_colors, alpha=io_type.seg_alpha)

    if ios.kps is not None:
        img_ = add_kp(img_, ios.kps)
    
    if ios.sn is not None:
        img_ = ios.sn

    return img_, text
