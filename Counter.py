import time

import re
def Counter(file = 'README.md'):
    """
    read .md or .txt format file
    :param file: .md or .txt format file
    :return: data
    """
    with open(file, 'r', encoding='UTF-8') as f:
        lines = f.readlines()
    num = 0
    for line in lines:
        p1 = re.compile(r'[{](.*?)[}]', re.S)  # 最小匹配
        if re.findall(p1, line):
            num += int(re.findall(p1, line)[0])

    return num
def write_num(file = 'README.md'):
    with open(file, 'a', encoding='UTF-8') as f:
        num_reference = Counter(file)
        run_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        str1 = 'I have already read ' + str(num_reference) + ' references so far (' + run_time + ').'
        f.write('\n')
        f.write(str1)
        f.write('\n')

write_num('README.md')
