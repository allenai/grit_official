import os
import cv2
import copy
import numpy as np
from PIL import Image, ImageDraw
from typing import List, Tuple, Optional
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
    N = len(masks)

    for i,mask in enumerate(masks):
        img[mask==1,:] = (1-alpha)*img[mask==1,:] + alpha*np.array([colors[i]])


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


def grit_viz(img,ios,io_type=GritIOType.PRED,text=None):
    img_ = copy.deepcopy(img)
    
    if text is None:
        text = dict()
    
    if ios.text is not None:
        text[io_type.prefix] = ios.text

    if ios.boxes is not None:
        add_bboxes(img_, ios.boxes, io_type.box_color, io_type.box_thickness)
    
    if ios.segs is not None:
        add_seg(img_, ios.segs, colors=io_type.seg_colors, alpha=io_type.seg_alpha)

    return img_, text
