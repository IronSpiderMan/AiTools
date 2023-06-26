import json

import cv2
import torch
from PIL import Image
from collections import namedtuple
from torchvision import transforms
from hubs.model import MattingNetwork


def segment_test():
    # 加载模型
    segmentor = MattingNetwork('mobilenetv3').eval()
    segmentor.load_state_dict(torch.load('../static/models/rvm_mobilenetv3.pth'))

    src = Image.open('../static/segments/001/e716048bd4ede0ab02440ec1f28f00c3.jpeg')
    src = (transforms.PILToTensor()(src) / 255.)[None]
    with torch.no_grad():
        fgr, pha, *rec = segmentor(src)
        segmented = torch.cat([src, pha], dim=1).squeeze(0).permute(1, 2, 0).numpy()
        segmented = cv2.normalize(segmented, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        Image.fromarray(segmented).show()


def namedtuple2json(nt):
    nt_json = {}
    for field in nt._fields:
        attr = getattr(nt, field)
        if isinstance(attr, (int, str, float)):
            nt_json[field] = attr
        else:
            nt_json[field] = None
    return nt_json


class Test:
    def __init__(self):
        self.a = 1
        self.b = 2


if __name__ == '__main__':
    t = Test()
    # namedtuple2json()
    # Result = namedtuple('Result', ['f1', 'f2'])
    # r1 = Result(1, 2)
    # print(getattr(r1, 'f1'))
