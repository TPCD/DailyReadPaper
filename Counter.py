import time
import os
import re
def counter(file = 'README.md'):
    """
    read .md or .txt format file
    :param file: .md or .txt format file
    :return: data
    """
    num = 0
    with open(file, 'r', encoding='UTF-8') as f:
        lines = f.readlines()

    for line in lines:
        p1 = re.compile(r'[{](.*?)[}]', re.S)  # 最小匹配
        if re.findall(p1, line):
            num += int(re.findall(p1, line)[0])

    return num
def write_num(file = 'README.md'):
    with open(file, 'a', encoding='UTF-8') as f:
        num_reference = counter(file)
        run_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        str1 = 'I have already read ' + str(num_reference) + ' references so far (' + run_time + ').'
        f.write('\n')
        f.write(str1)
        f.write('\n')

def extract_reference(month_path = os.path.join( os.getcwd(), '2019', '02')):
    g = os.walk(month_path)
    references = []
    n = 0
    for path, dir_list, file_list in g:
        for file_name in file_list:
            if file_name == 'Papers.md':
                with open(os.path.join(path, file_name), 'r', encoding='UTF-8') as f:
                    content = f.read()
                    p1 = re.compile(r'>((?:.|\n)*?)[}][\n][}]', re.X)  # 最小匹配
                    p2 = re.compile(r'>((?:.|\n)*?)}', re.X)  # 最小匹配

                    reference = re.findall(p1, content)
                    if reference:
                        for r in reference:
                            print(r + '}\n}')
                            references.append(r + '}\n}')
                            n += 1
    print(n)
    print(len(references))

def extract_cite(month_path = os.path.join( os.getcwd(), '2019', '02')):
    g = os.walk(month_path)
    cites = []
    n = 0
    for path, dir_list, file_list in g:
        for file_name in file_list:
            if file_name == 'Papers.md':
                with open(os.path.join(path, file_name), 'r', encoding='UTF-8') as f:
                    content = f.read()
                    p1 = re.compile(r'>((?:.|\n)*?)[}][\n][}]', re.X)  # 最小匹配
                    p2 = re.compile(r'>((?:.|\n)*?)}', re.X)  # 最小匹配

                    reference = re.findall(p1, content)
                    p3 = re.compile(r'{(?:.)*?,')

                    if reference:
                        for r in reference:
                            author = re.search(p3, r).group()
                            author = author.replace(',', '}')
                            author = '\cite' + author
                            print(author)
                            cites.append(author)
                            n += 1
    print(n)
    print(len(cites))
# write_num('README.md')

if __name__ == '__main__':
    write_num('README.md')
    # extract_reference('./2019/02')
    # extract_cite('./2019/02')