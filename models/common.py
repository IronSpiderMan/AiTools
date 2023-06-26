from collections import namedtuple

"""
上传图片返回的结果信息
"""
UploadResult = namedtuple('UploadResult', ['status', 'msg', 'url'])
SegmentResult = UploadResult
