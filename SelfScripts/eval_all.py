# -*-coding:utf-8-*-

import os
import caffe
import sys
import cv2
import time
import numpy as np
import argparse
from PIL import Image

from data_clr import cityscapes_19_label

caffe_root = '/opt/PSPNET-cudnn5/' 			# Change this to the absolute directoy to SegNet Caffe
sys.path.insert(0, caffe_root + 'python')


class EvalDataSet:
    def __init__(self, data_name=''):
        os.environ["CUDA_VISIBLE_DEVICES"] = '3'
        caffe.set_mode_gpu()

        if data_name == '':
            data_name = 'cityscapes'
        self.data_name = data_name
        self.net = None

        if self.data_name == 'ADE20K':
            self.isVal = True  # evaluation on valset
            self.step = 2000  # equals to number of images divide num of GPUs in testing e.g. 500=2000/4
            self.data_root = '/data/deeplearning/dataset'  # root path of dataset
            # evaluation list, refer to lists in folder 'samplelist'
            self.eval_list = '/opt/PSPNet/data/ADE20K/list/ADE20K_val.txt'
            self.save_root = '/opt/PSPNet/data/ADE20K/mc_result/ADE20K/val/pspnet50_473/'  # root path to store the result image
            self. model_weights = '/opt/PSPNet/data/ADE20K/model/pspnet50_ADE20K.caffemodel'
            self.model_deploy = '/opt/PSPNet/mpspnet50_ADE20K_473.prototxt'
            self.fea_cha = 150  # number of classes
            self.base_size = 512  # based size for scaling
            self.crop_size = 473  # crop size fed into network
            self.data_class = '/opt/PSPNet/data/ADE20K/objectName150.mat'  # class name
            self.data_colormap = '/opt/PSPNet/data/ADE20K/color150.mat'  # color map
            self.colours = '/opt/PSPNet/data/ADE20K/clr.png'
        elif self.data_name == 'VOC2012':
            self.isVal = False  # evaluation on testset
            self.step = 1456  # 364=1456/4
            self.data_root = '/data/deeplearning/dataset'
            self.eval_list = '/opt/PSPNet/data/VOC2012/list/VOC2012_test.txt'
            self.save_root = '/opt/PSPNet/data/VOC2012/mc_result/VOC2012/test/pspnet101_473/'
            self.model_weights = '/opt/PSPNet/data/VOC2012/model/pspnet101_VOC2012.caffemodel'
            self.model_deploy = '/opt/PSPNet/data/VOC2012/prototxt/pspnet101_VOC2012_473.prototxt'
            self.fea_cha = 21
            self.base_size = 512
            self.crop_size = 473
            self.data_class = '/opt/PSPNet/data/VOC2012/objectName21.mat'
            self.data_colormap = '/opt/PSPNet/data/VOC2012/colormapvoc.mat'
            self.colours = '/opt/PSPNet/data/VOC2012/clr.png'
        elif self.data_name == 'cityscapes':
            self.isVal = True
            self.step = 500  # 125=500/4
            self.data_root = '/data/deeplearning/dataset/sample'
            # self.eval_list = '/opt/PSPNet/data/cityscapes/list/cityscapes_val.txt'
            # self.eval_list = '/data/deeplearning/dataset/test0928/test0928.txt'
            # self.save_root = '/data/deeplearning/dataset/test0928/pspnet/cityscapes/'
            self.eval_list = '/data/deeplearning/dataset/sample/sample.txt'
            self.save_root = '/data/deeplearning/dataset/sample/pspnet/cityscapes/'
            self.model_weights = '/opt/PSPNet/data/cityscapes/model/pspnet101_cityscapes.caffemodel'
            self.model_deploy = '/opt/PSPNet/data/cityscapes/prototxt/pspnet101_cityscapes_713.prototxt'
            self.fea_cha = 19
            self.base_size = 2048
            self.crop_size = 713
            self.data_class = '/opt/PSPNet/data/cityscapes/objectName19.mat'
            self.data_colormap = '/opt/PSPNet/data/cityscapes/colormapcs.mat'
            self.colours = '/opt/PSPNet/data/cityscapes/clr.png'

        # skip serveal images in the list
        self.skipsize = 0

        self.is_save_feat = False  # set to true if final feature map is needed (not suggested for storage consuming)
        self.save_gray_folder = self.save_root + 'gray/'  # path for predicted gray image
        self.save_color_folder = self.save_root + 'color/'  # path for predicted color image
        self.save_feat_folder = self.save_root + 'feat/'  # path for predicted feature map
        self.scale_array = [1]  # set to [0.5 0.75 1 1.25 1.5 1.75] for multi-scale testing
        self.mean_r = 123.68  # means to be subtracted and the given values are used in our training stage
        self.mean_g = 116.779
        self.mean_b = 103.939

        self.acc = 0
        self.iou = 0
        # multi-GPUs for parfor testing, if number of GPUs is changed, remember to change the variable 'step'
        self.gpu_id_array = [0, 1, 2, 3]
        self.runID = 1
        self.gpu_num = 1

    def run(self):
        self.eval_sub()

        if self.isVal:
            self.eval_acc()
        return

    def eval_sub(self):
        if not os.path.exists(self.save_gray_folder):
            os.makedirs(self.save_gray_folder)

        if not os.path.exists(self.save_color_folder):
            os.makedirs(self.save_color_folder)

        if not os.path.exists(self.save_feat_folder):
            os.makedirs(self.save_feat_folder)

        if not os.path.exists(self.model_weights):
            print("Model weights file is not exist!")
            return

        if not os.path.exists(self.model_deploy):
            print("Model prototxt file is not exist!")
            return

        if not os.path.exists(self.eval_list):
            print("eval file is not exist!")
            return

        file_list = []
        with open(self.eval_list, "rb") as f:
            file_name = f.readline()
            while file_name:
                file_list.append(file_name)
                file_name = f.readline()

        if self.net is None:
            self.net = caffe.Net(self.model_deploy,
                                 self.model_weights,
                                 caffe.TEST)

        input_shape = self.net.blobs['data'].data.shape
        label_colours = cv2.imread(self.colours).astype(np.uint8)

        for id_ in file_list:
            # [image, label] = str(id_).strip().split(" ")
            image = str(id_).strip('\n')
            file_path = os.path.join(self.data_root, image)
            start = time.time()

            origin_frame = cv2.imread(file_path, cv2.IMREAD_COLOR)

            width = origin_frame.shape[1]
            height = origin_frame.shape[0]

            # input_shape[3]=width, input_shape[2]=height
            frame = cv2.resize(origin_frame, (input_shape[3], input_shape[2]))
            input_image = frame.transpose((2, 0, 1))
            input_image = np.asarray([input_image])
            # print(self.net.inputs)
            # print(self.net.blobs)
            self.net.forward_all(data=input_image)

            predict = self.net.blobs['conv6_interp'].data
            out_pred = np.resize(predict, (3, input_shape[2], input_shape[3]))
            out_pred = out_pred.transpose(1, 2, 0).astype(np.uint8)
            for j in range(0, 713):
                for k in range(0, 713):
                    x = -1
                    label = 0
                    for i in range(0, 19):
                        if predict[0][i][j][k] > x:
                            x = predict[0][i][j][k]
                            label = i
                    out_pred[j][k][0] = out_pred[j][k][1] = out_pred[j][k][2] = label
            out_rgb = np.zeros(out_pred.shape, dtype=np.uint8)

            cv2.LUT(out_pred, label_colours, out_rgb)
            rgb_frame = cv2.resize(out_rgb, (width, height), interpolation=cv2.INTER_NEAREST)

            file_name_path = str(image).split("/")
            file_name = file_name_path[len(file_name_path) - 1]
            file_name_only = (str(file_name).split("."))[0]
            file_name = file_name_only + ".png"
            out_path = os.path.join(self.save_color_folder, file_name)
            cv2.imwrite(out_path, rgb_frame)

            # pred_data = self.net.blobs['conv6_interp'].data
            # output = np.squeeze(self.net.blobs['conv6_interp'].data)
            # ind = np.argmax(output, axis=1)
            #
            # segmentation_ind = np.squeeze(self.net.blobs['conv6_interp'].data)
            # segmentation_ind_3ch = np.resize(segmentation_ind, (3, input_shape[2], input_shape[3]))
            # # segmentation_ind_3ch = np.resize(segmentation_ind, (3, height, width))
            # segmentation_ind_3ch = segmentation_ind_3ch.transpose(1, 2, 0).astype(np.uint8)
            # segmentation_rgb = np.zeros(segmentation_ind_3ch.shape, dtype=np.uint8)
            #
            # cv2.LUT(segmentation_ind_3ch, label_colours, segmentation_rgb)
            #
            # rgb_frame = cv2.resize(segmentation_rgb, (width, height), interpolation=cv2.INTER_NEAREST)
            #
            # file_name_path = str(image).split("/")
            # file_name = file_name_path[len(file_name_path) - 1]
            # result_path = os.path.join(self.save_color_folder, file_name)
            # cv2.imwrite(result_path, rgb_frame)

            end = time.time()
            print("Processed {} in {} ms\n".format(file_path ,str((end - start) * 1000)))

        return

    def eval_acc(self):
        if not os.path.exists(self.data_class):
            print("data class file is not exist!")
        return

    def generate_clr(self):
        label_clr = {}

        if self.data_name == 'cityscapes':
            label_clr = {l.id: l.color for l in cityscapes_19_label}
        elif self.data_name == 'ADE20K':
            label_clr = {}

        clr_image = self.colours
        width = 256
        height = 1
        image = Image.new("RGBA", (width, height), (0, 0, 0))
        image_data = image.load()

        class_count = len(label_clr)
        for i in range(class_count):
            image_data[i, 0] = label_clr[i]

        image.save(clr_image)
        return


if __name__ == '__main__':

    print("start\n")
    time1 = time.time()

    eval_ = EvalDataSet(data_name='cityscapes')
    # eval_.generate_clr()

    eval_.run()

    time2 = time.time()
    print("finish in {} s\n".format(time2 - time1))
