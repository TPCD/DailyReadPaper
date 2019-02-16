import numpy as np
import re
def Counter(file):
    """
    read .md or .txt format file
    :param file: .md or .txt format file
    :return: data
    """
    with open('README.md', 'r', encoding='UTF-8') as f:
        lines = f.readlines()
    num = 0
    for line in lines:
        p1 = re.compile(r'[{](.*?)[}]', re.S)  # 最小匹配
        if re.findall(p1, line):
            num += int(re.findall(p1, line)[0])

    return num

num_reference = Counter('README.md')
print(num_reference)
