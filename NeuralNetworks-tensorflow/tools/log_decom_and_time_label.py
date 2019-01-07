#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""Decompress log files and label time.


Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    2019/1/6 11:30
"""
import os
import bz2
import shutil

local_path = os.getcwd()
print(local_path)

for (dir_path, dir_names, files) in os.walk(local_path):
    print("########Current dir:{0}########".format(dir_path))
    print("Local dir names:" + str(dir_names))
    print("Local file names:" + str(files))
    for file in files:
        if file.endswith(".tar.bz2"):
            file_path = os.path.join(dir_path, file)
            file_name = file.split(".tar")[0]
            unzip_file_path = os.path.join(dir_path, file_name)

            # Unzip log tar file.
            print("Unzip the {0} to {1} ...".format(file_path, unzip_file_path))
            try:
                with open(unzip_file_path, "wb") as new_file, bz2.BZ2File(
                        file_path, "rb") as file:
                    for data in iter(lambda: file.read(100 * 1024), b''):
                        new_file.write(data)
            except EOFError as e:
                print(e)
            except Exception as e:
                print(e)
                continue

            # Label end time for logcat log and kernel log.
            try:
                if file_name.startswith("logcat_log"):
                    with open(unzip_file_path, "rb") as log_file:
                        lines = log_file.readlines()
                        if len(lines) == 0:
                            os.remove(unzip_file_path)  # File is empty.
                            continue
                        time_flag = lines[-2].split(b" ")[: 2]
                        time_flag = time_flag[0].replace(b"-", b"_") + b"_" + \
                                    time_flag[1].split(
                                        b".")[0].replace(b":", b"_")
                        time_flag = time_flag.decode("utf-8")
                        time_label_file_path = os.path.join(dir_path,
                                                            time_flag +
                                                            "_end_time_" +
                                                            file_name)
                        print("Move the {0} to {1} ...".format(unzip_file_path,
                                                               time_label_file_path))
                    shutil.move(unzip_file_path, time_label_file_path)
                elif file_name.startswith("kernel_log"):
                    with open(unzip_file_path, "rb") as log_file:
                        lines = log_file.readlines()
                        if len(lines) == 0:
                            os.remove(unzip_file_path)  # File is empty.
                            continue
                        time_flag = lines[-2].split(b" ")[1: 5]
                        if time_flag[1] == b"":
                            time_flag = time_flag[0] + b"_" + time_flag[
                                2] + b"_" + \
                                        time_flag[3].replace(b":", b"_")
                        else:
                            time_flag = time_flag[0] + b"_" + time_flag[
                                1] + b"_" + \
                                        time_flag[2].replace(b":", b"_")
                        time_flag = time_flag.decode("utf-8")
                        time_label_file_path = os.path.join(dir_path,
                                                            time_flag +
                                                            "_end_time_" +
                                                            file_name)
                        print("Move the {0} to {1} ...".format(unzip_file_path,
                                                               time_label_file_path))
                    shutil.move(unzip_file_path, time_label_file_path)
            except Exception as e:
                os.remove(unzip_file_path)  # File is empty.
                print(e)
                continue
        else:
            # Label end time for logcat log and kernel log.
            try:
                file_name = file
                file_path = os.path.join(dir_path, file_name)
                if file_name.startswith("logcat_log"):
                    with open(file_path, "rb") as log_file:
                        lines = log_file.readlines()
                        time_flag = lines[-1].split(b" ")[: 2]
                        time_flag = time_flag[0].replace(b"-", b"_") + b"_" + \
                                    time_flag[1].split(
                                        b".")[0].replace(b":", b"_")
                        time_flag = time_flag.decode("utf-8")
                        time_label_file_path = os.path.join(dir_path,
                                                            time_flag +
                                                            "_end_time_" +
                                                            file_name)
                        print("Copy the {0} to {1} ...".format(file_path,
                                                               time_label_file_path))
                    shutil.copy(file_path, time_label_file_path)
                elif file_name.startswith("kernel_log"):
                    with open(file_path, "rb") as log_file:
                        lines = log_file.readlines()
                        time_flag = lines[-1].split(b" ")[1: 5]
                        if time_flag[1] == b"":
                            time_flag = time_flag[0] + b"_" + time_flag[
                                2] + b"_" + \
                                        time_flag[3].replace(b":", b"_")
                        else:
                            time_flag = time_flag[0] + b"_" + time_flag[
                                1] + b"_" + \
                                        time_flag[2].replace(b":", b"_")
                        time_flag = time_flag.decode("utf-8")
                        time_label_file_path = os.path.join(dir_path,
                                                            time_flag +
                                                            "_end_time_" +
                                                            file_name)
                        print("Copy the {0} to {1} ...".format(file_path,
                                                               time_label_file_path))
                    shutil.copy(file_path, time_label_file_path)
            except Exception as e:
                print(e)
                continue
