# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
#
"""The download tool to download image net data.


Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    2018/11/12 10:54
"""
import urllib.request as req
import os
import threading
import math

image_data_dir = "../image_net_data/"


def download_images(urls_path, save_path, label, n_thread):
    """Download images by urls file.

    Args:
        urls_path: The urls file path.
        save_path: The save dir path.
        label: The label of these images.
        n_thread: The quantity of threads used to download.
    """

    # Define the download function.
    def download(urls, index):
        for url in urls:
            try:
                print("Download ", index)
                file_path = save_path + label + "_" + str(index) + ".jpg"
                req.urlretrieve(url, file_path)
                print(str(index) + " success!")
                index += 1
            except Exception as e:
                print(str(index) + " failed:" + str(e))
                pass

    file = open(urls_path, "r", encoding='UTF-8')
    lines = file.readlines()
    lenth = len(lines)
    batch_size = math.ceil(lenth / n_thread)

    print("Download begin.")
    # Start all threads.
    threads = []
    for index_thread in range(n_thread):
        print("Download thread {0}-{1}".format(index_thread * batch_size,
                                               (index_thread + 1) * batch_size))
        t = threading.Thread(target=download,
                             args=(lines[index_thread * batch_size: (
                                 index_thread + 1) * batch_size],
                                   index_thread * batch_size))
        t.start()
        threads.append(t)
    # Wait for all threads.
    for t in threads:
        print("wait threads", str(t))
        t.join()
    print("Download end.")
    file.close()


def download_all(urls_dir_path, n_thread):
    """Download all images by every urls files in the dir.

    Args:
        urls_dir_path: The urls files dir path.
        n_thread: The quantity of threads used to download.
    """
    filesList = os.listdir(urls_dir_path)
    for file in filesList:
        file_name = file.split(".")[0]
        print(file_name)
        if not os.path.exists(image_data_dir + file_name):
            os.makedirs(image_data_dir + file_name)
        else:
            os.remove(image_data_dir + file_name)
        download_images(urls_path=urls_dir_path + file,
                        save_path=image_data_dir + file_name + "/",
                        label=file_name,
                        n_thread=n_thread)


if __name__ == "__main__":
    download_all(image_data_dir, 32) # Use 32 treads to accelerate the download.
