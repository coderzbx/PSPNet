import caffe
import numpy as np
import cv2
import os
import time
import argparse


class ModelSegNetDemo:
    def __init__(self, model, weights, colours, gpu_id=3):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        caffe.set_mode_gpu()

        self.weights = weights
        self.model = model
        self.colours = colours

        self.net = caffe.Net(self.model,
                        self.weights,
                        caffe.TEST)

    def do(self, image_data):

        input_shape = self.net.blobs['data'].data.shape
        label_colours = cv2.imread(self.colours).astype(np.uint8)

        start = time.time()

        image = np.asarray(bytearray(image_data), dtype="uint8")
        origin_frame = cv2.imdecode(image, cv2.IMREAD_COLOR)

        width = origin_frame.shape[1]
        height = origin_frame.shape[0]

        frame = cv2.resize(origin_frame, (input_shape[3], input_shape[2]))
        input_image = frame.transpose((2, 0, 1))
        input_image = np.asarray([input_image])
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

        img_array = cv2.imencode('.png', rgb_frame)
        img_data = img_array[1]
        pred_data = img_data.tostring()

        end = time.time()
        print('%30s' % 'Processed results in ', str((end - start) * 1000), 'ms\n')

        return pred_data


if __name__ == '__main__':

    weights = ''
    model = ''
    colours = ''

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=False)
    parser.add_argument('--weights', type=str, required=False)
    parser.add_argument('--colours', type=str, required=False)

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--file', type=str, required=False)
    group.add_argument('--dir', type=str, required=False)

    parser.add_argument('--gpu', type=str, required=False)
    args = parser.parse_args()

    if args.model and args.model != '' and os.path.exists(args.model):
        model = args.model
        print(model)

    if not os.path.exists(model):
        print("model file [{}] is not exist\n".format(model))
        exit(1)

    if args.weights and args.weights != '' and os.path.exists(args.weights):
        weights = args.weights
        print(weights)

    if not os.path.exists(weights):
        print("weights file [{}] is not exist\n".format(weights))
        exit(1)

    if args.colours and args.colours != '' and os.path.exists(args.colours):
        colours = args.colours
        print(colours)

    if not os.path.exists(colours):
        print("colours file [{}] is not exist\n".format(colours))
        exit(1)

    procFile = False
    file_path = ''
    if args.file and args.file != '' and os.path.exists(args.file):
        procFile = True
        file_path = args.file

    procDir = False
    file_dir = ''
    if args.dir and args.dir != '' and os.path.exists(args.dir):
        procDir = True
        file_dir = args.dir

    if procFile and not os.path.exists(file_path):
        print("image file [{}] is not exist\n".format(file_path))
        exit(1)

    if procDir and not os.path.exists(file_dir):
        print("image dir [{}] is not exist\n".format(file_dir))
        exit(1)

    gpu_id = 0
    if args.gpu:
        gpu_id = args.gpu

    seg_model = ModelSegNetDemo(model=model, weights=weights, colours=colours, gpu_id=gpu_id)

    if procDir:
        result_dir = os.path.join(file_dir, 'pspnet')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        origin_list = os.listdir(file_dir)

        for _image in origin_list:
            image_path = os.path.join(file_dir, _image)
            name_list = _image.split('.')
            if (len(name_list) < 2):
                print(image_path)
                continue
            file_name = name_list[0]
            ext_name = name_list[1]
            if ext_name == 'jpg' or ext_name == 'png':
                recog_path = os.path.join(result_dir, file_name + '.png')
                with open(image_path, 'rb') as f:
                    image_data = f.read()
                    recog_data = seg_model.do(image_data=image_data)

                    with open(recog_path, 'wb') as w:
                        w.write(recog_data)

    if procFile:
        name_list = file_path.split('/')
        part_count = len(name_list)
        if part_count < 2:
            exit(0)

        file_name = name_list[part_count - 1]
        name_len = len(file_name)

        file_dir = file_path[:(-1)*name_len]
        result_dir = os.path.join(file_dir, 'results')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        name_list = file_name.split('.')
        if (len(name_list) < 2):
            print(file_path)

        file_name = name_list[0]
        ext_name = name_list[1]
        if ext_name == 'jpg' or ext_name == 'png':
            recog_path = os.path.join(result_dir, file_name + '.png')
            with open(file_path, 'rb') as f:
                image_data = f.read()
                recog_data = seg_model.do(image_data=image_data)

                with open(recog_path, 'wb') as w:
                    w.write(recog_data)
