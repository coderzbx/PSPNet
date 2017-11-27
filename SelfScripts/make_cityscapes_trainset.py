# -*-coding:utf-8-*-

import argparse
import os
import time
import shutil

# src_dir：cityscapes散列的数据
# dest_dir：统一存放数据的目录


class CityScapeDataset:
    def __init__(self):
        return

    def transform_image(self, src_dir, dest_dir):
        image_list = os.listdir(src_dir)
        for id_ in image_list:
            dir = os.path.join(src_dir, id_)
            if os.path.isdir(dir):
                list1 = os.listdir(dir)
                for id1 in list1:
                    name_list = str(id1).split(".")
                    name_ext = name_list[len(name_list) - 1]
                    name = name_list[0]
                    name_list = name.split("_")
                    name_list = name_list[:-1]
                    name = ''
                    for name_part in name_list:
                        name += name_part + "_"
                    name = name.rstrip("_")
                    name += "." + name_ext
                    id_path = os.path.join(dir, id1)
                    dest_image_path = os.path.join(dest_dir, name)
                    shutil.copyfile(id_path, dest_image_path)
            else:
                src_image_path = dir

                name_list = str(id_).split(".")
                name_ext = name_list[len(name_list) - 1]
                name = name_list[0]
                name_list = name.split("_")
                name_list = name_list[:-1]
                name = ''
                for name_part in name_list:
                    name += name_part + "_"
                name = name.rstrip("_")
                name += "." + name_ext

                dest_image_path = os.path.join(dest_dir, name)
                shutil.copyfile(src_image_path, dest_image_path)
        return

    def transform_label(self, src_dir, dest_dir):
        image_list = os.listdir(src_dir)
        for id_ in image_list:
            dir = os.path.join(src_dir, id_)
            if os.path.isdir(dir):
                list1 = os.listdir(dir)
                for id1 in list1:
                    if str(id1).find("gtFine_color") == -1:
                        continue

                    name_list = str(id1).split(".")
                    name_ext = name_list[len(name_list) - 1]
                    name = name_list[0]
                    name_list = name.split("_")
                    name_list = name_list[:-2]
                    name = ''
                    for name_part in name_list:
                        name += name_part + "_"
                    name = name.rstrip("_")
                    name += "." + name_ext
                    id_path = os.path.join(dir, id1)
                    dest_image_path = os.path.join(dest_dir, name)
                    shutil.copyfile(id_path, dest_image_path)
            else:
                if str(id_).find("gtFine_color") == -1:
                    continue
                src_image_path = dir

                name_list = str(id_).split(".")
                name_ext = name_list[len(name_list) - 1]
                name = name_list[0]
                name_list = name.split("_")
                name_list = name_list[:-1]
                name = ''
                for name_part in name_list:
                    name += name_part + "_"
                name = name.rstrip("_")
                name += "." + name_ext

                dest_image_path = os.path.join(dest_dir, name)
                shutil.copyfile(src_image_path, dest_image_path)
        return

    def transform_annotation(self, src_dir, dest_dir):
        image_list = os.listdir(src_dir)
        for id_ in image_list:
            dir = os.path.join(src_dir, id_)
            if os.path.isdir(dir):
                list1 = os.listdir(dir)
                for id1 in list1:
                    if str(id1).find("gtFine_labelIds") == -1:
                        continue

                    name_list = str(id1).split(".")
                    name_ext = name_list[len(name_list) - 1]
                    name = name_list[0]
                    name_list = name.split("_")
                    name_list = name_list[:-2]
                    name = ''
                    for name_part in name_list:
                        name += name_part + "_"
                    name = name.rstrip("_")
                    name += "." + name_ext
                    id_path = os.path.join(dir, id1)
                    dest_image_path = os.path.join(dest_dir, name)
                    shutil.copyfile(id_path, dest_image_path)
            else:
                if str(id_).find("gtFine_labelIds") == -1:
                    continue
                src_image_path = dir

                name_list = str(id_).split(".")
                name_ext = name_list[len(name_list) - 1]
                name = name_list[0]
                name_list = name.split("_")
                name_list = name_list[:-1]
                name = ''
                for name_part in name_list:
                    name += name_part + "_"
                name = name.rstrip("_")
                name += "." + name_ext

                dest_image_path = os.path.join(dest_dir, name)
                shutil.copyfile(src_image_path, dest_image_path)
        return


if __name__ == '__main__':
    time1 = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, required=True)
    parser.add_argument('--dest_dir', type=str, required=True)
    parser.add_argument('--type', type=str, required=True)
    args = parser.parse_args()

    src_dir = args.src_dir
    dest_dir = args.dest_dir

    transFormer = CityScapeDataset()

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    proc_type = args.type
    if proc_type == 'image':
        transFormer.transform_image(src_dir=src_dir, dest_dir=dest_dir)
    elif proc_type == 'label':
        transFormer.transform_label(src_dir=src_dir, dest_dir=dest_dir)
    elif proc_type == 'annot':
        transFormer.transform_annotation(src_dir=src_dir, dest_dir=dest_dir)

    time2 = time.time()

    print("finish in {} s\n".format(time2 - time1))