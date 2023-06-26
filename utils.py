import hashlib

SALT = "Zack Fair"


def str2md5(dirname):
    """
    对字符串进行md5加密
    :param dirname: 需要加密的字符串
    :return:
    """
    hl = hashlib.md5()
    hl.update(SALT.encode(encoding="utf-8"))
    hl.update(dirname.encode(encoding="utf-8"))
    return hl.hexdigest()


if __name__ == '__main__':
    print(str2md5('dfsadf'))
