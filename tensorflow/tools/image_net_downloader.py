# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
#
"""The download tool to download image net.


Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    2018/11/12 10:54
"""
import urllib.request as req
import os

image_data_dir = "../image_net_data/"

def download_images(urls_path, save_path, label):
    """Download images by urls file.

    Args:
        urls_path: The urls file path.
        save_path: The save dir path.
        label: The label of these images.
    """
    file = open(urls_path)
    index = 0
    for line in file:
        try:
            print("Download ", line)
            file_path = save_path + label + "_" + str(index) + ".jpg"
            req.urlretrieve(line, file_path)
            index += 1
            print("success")
        except Exception as e:
            print(str(e))
            pass
    file.close()


def download_all(urls_dir_path):
    """Download all images by every urls files in the dir.

    Args:
        urls_dir_path: The urls files dir path.
    """
    filesList = os.listdir(urls_dir_path)
    for file in filesList:
        file_name = file.split(".")[0]
        print(file_name)
        if not os.path.exists(image_data_dir + file_name):
            os.makedirs(image_data_dir + file_name)
        download_images(urls_path=urls_dir_path + file,
                        save_path=image_data_dir + file_name + "/",
                        label=file_name)


if __name__ == "__main__":
    download_all(image_data_dir)