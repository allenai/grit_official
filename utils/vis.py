import os
import cv2
import copy
import base64
import torch
import numpy as np
from io import BytesIO
from PIL import Image, ImageDraw
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass, fields, field
from torchvision.utils import draw_keypoints

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


def add_kp(img: np.ndarray, kps: List[list]) -> np.ndarray:
    img = torch.from_numpy(img)
    img = torch.einsum('hwc->chw',img)
    for i,kp in enumerate(kps):
        k_array3d = np.reshape(np.array(kp),(N_KEYPOINTS,N_DIM))
        kp_xy = k_array3d[:,:2]
        k_vis = k_array3d[:,2]

        if k_vis.sum() == 0:
            # no keypoints are labeled for this person
            continue 

        visible_kps = np.argwhere(k_vis).ravel().tolist()
        connections = [c for c in CONNECT_SKELETON if set(c).issubset(set(visible_kps))]

        color = RGB_COLORS[i]
        kp_torch = torch.from_numpy(np.expand_dims(kp_xy, axis=0))
        img = draw_keypoints(img, kp_torch, connectivity=connections, colors=color, radius=5, width=4)
    
    img = torch.einsum('chw->hwc',img)
    return img.detach().numpy()


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
