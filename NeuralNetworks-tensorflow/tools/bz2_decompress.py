#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
"""The bz2 decompress tool.


Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    2018/12/6 11:59
"""

import os
import bz2

local_path = os.getcwd()
print(local_path)

for (dir_path, dir_names, files) in os.walk(local_path):
    print("########Current dir:{0}########".format(dir_path))
    print("Local dir names:" + str(dir_names))
    print("Local file names:" + str(files))
    for file in files:
        if file.endswith(".tar.bz2"):
            file_path = os.path.join(dir_path, file)
            new_file_path = os.path.join(dir_path, file.split(".tar")[0] +
                                         ".log")
            print("Unzip the {0} to {1} ...".format(file_path, new_file_path))
            with open(new_file_path, "wb") as new_file, bz2.BZ2File(
                    file_path, "rb") as file:
                for data in iter(lambda: file.read(100 * 1024), b''):
                    new_file.write(data)
        else:
            print("Not bz2")
