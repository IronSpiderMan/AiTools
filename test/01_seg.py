import cv2

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from hubs.inference_utils import ImageSequenceReader
from hubs.model import MattingNetwork

# 加载模型
segmentor = MattingNetwork('mobilenetv3').eval()
segmentor.load_state_dict(torch.load('../static/models/rvm_mobilenetv3.pth'))

# source = ImageSequenceReader('../static/segments/001', transforms.ToTensor())
# reader = DataLoader(source, batch_size=12, pin_memory=True)
# with torch.no_grad():
#     for src in reader:
#         print(src)
#         print(src.shape)
#         fgr, pha, *rec = segmentor(src)
#         segmented = torch.cat([src, pha], dim=1).squeeze(0).permute(1, 2, 0).numpy()
#         segmented = cv2.normalize(segmented, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
#         Image.fromarray(segmented).show()

src = Image.open('../static/segments/001/e716048bd4ede0ab02440ec1f28f00c3.jpeg')
src = (transforms.PILToTensor()(src) / 255.)[None]
with torch.no_grad():
    fgr, pha, *rec = segmentor(src)
    segmented = torch.cat([src, pha], dim=1).squeeze(0).permute(1, 2, 0).numpy()
    segmented = cv2.normalize(segmented, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    Image.fromarray(segmented).show()
