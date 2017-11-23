import caffe
import numpy as np
import cv2
import os
import json
import time
import argparse
import requests
from PIL import Image
from io import BytesIO
import multiprocessing
from multiprocessing import Manager


class DownloadTask:
    def __init__(self, track_point_id, exit_flag=False):
        self.track_point_id = track_point_id
        self.exit_flag = exit_flag


class RecogTask:
    def __init__(self, track_point_id, image_data, exit_flag=False):
        self.track_point_id = track_point_id
        self.image_data = image_data
        self.exit_flag = exit_flag


class SaveTask:
    def __init__(self, track_point_id, pred_data, exit_flag=False):
        self.track_point_id = track_point_id
        self.pred_data = pred_data
        self.exit_flag = exit_flag


class ModelDemo:
    def __init__(self, model, weights, colours, manager, gpu_id=3):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        caffe.set_mode_gpu()

        self.manager = manager
        self.download_queue = self.manager.Queue()
        self.recog_queue = self.manager.Queue()
        self.save_queue = self.manager.Queue()

        self.weights = weights
        self.model = model
        self.colours = colours

        self.net = caffe.Net(self.model,
                        self.weights,
                        caffe.TEST)

    def start_queue(self, url, track_id):
        _url = url + "/track/get"

        try:
            res = requests.post(url=_url, data={'trackId': track_id})
            track_info = res.text

            track_data = json.loads(track_info)
            code = track_data["code"]

            if code != "0":
                return False

            point_data = track_data["result"]["pointList"]

            for point in point_data:
                track_point_id = point["trackPointId"]
                next_task = DownloadTask(track_point_id=track_point_id, exit_flag=False)
                self.download_queue.put(next_task)

        except Exception as e:
            print(e.args[0])

        next_task = DownloadTask(track_point_id=None, exit_flag=True)
        self.download_queue.put(next_task)
        next_task = DownloadTask(track_point_id=None, exit_flag=True)
        self.download_queue.put(next_task)

        return

    def download(self, url):
        if self.download_queue.empty():
            time.sleep(1)

        while True:
            task = self.download_queue.get()
            if not isinstance(task, DownloadTask):
                break

            if task.exit_flag:
                next_task = RecogTask(track_point_id=None, image_data=None, exit_flag=True)
                self.recog_queue.put(next_task)
                break

            image_data = None

            _url = url + "image/get"
            data = {
                "trackPointId": task.track_point_id,
                "type": "00",
                "seq": "004",
                "imageType": "jpg"
            }
            try:
                res_data = requests.post(url=_url, data=data)
                i = Image.open(BytesIO(res_data.content))
                output = BytesIO()
                i.save(output, format='JPEG')

                image_data = output.getvalue()
            except Exception as e:
                print e.args[0]

            next_task = RecogTask(track_point_id=task.track_point_id, image_data=image_data)
            self.recog_queue.put(next_task)

        return

    def recognition(self):
        if self.recog_queue.empty():
            time.sleep(1)

        while True:
            task = self.recog_queue.get()
            if not isinstance(task, RecogTask):
                break

            if task.exit_flag:
                next_task = SaveTask(track_point_id=None, pred_data=None, exit_flag=True)
                self.save_queue.put(next_task)
                next_task = SaveTask(track_point_id=None, pred_data=None, exit_flag=True)
                self.save_queue.put(next_task)

                break

            if task.image_data is None:
                continue

            pred_data = self.do(image_data=task.image_data)
            next_task = SaveTask(track_point_id=task.track_point_id, pred_data=pred_data)
            self.save_queue.put(next_task)
        return

    def save(self, dir):
        if self.save_queue.empty():
            time.sleep(1)

        while True:
            task = self.save_queue.get()
            if not isinstance(task, SaveTask):
                break

            if task.exit_flag:
                break

            file_name = task.track_point_id + ".png"
            recog_path = os.path.join(dir, file_name)

            with open(recog_path, 'wb') as w:
                w.write(task.pred_data)

        return

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

        predict = self.net.blobs['conv6_interp'].data[0, :, :, :]
        ind = np.argmax(predict, axis=0)
        out_pred = np.resize(ind, (3, input_shape[2], input_shape[3]))
        out_pred = out_pred.transpose(1, 2, 0).astype(np.uint8)
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

    time1 = time.time()

    weights = ''
    model = ''
    colours = ''

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--colours', type=str, required=True)
    parser.add_argument('--track_id', type=str, required=True)
    parser.add_argument('--url', type=str, required=True)
    parser.add_argument('--dir', type=str, required=True)
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

    gpu_id = 0
    if args.gpu:
        gpu_id = args.gpu

    save_dir = args.dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    manager = Manager()
    seg_model = ModelDemo(model=model, weights=weights, colours=colours, manager=manager, gpu_id=gpu_id)

    seg_model.start_queue(url=args.url, track_id=args.track_id)
    seg_model.download(url=args.url)
    seg_model.recognition()
    seg_model.save(dir=args.dir)

    # all_process = []
    #
    # download_process1 = multiprocessing.Process(target=seg_model.download, args=(args.url,))
    # download_process2 = multiprocessing.Process(target=seg_model.download, args=(args.url,))
    # all_process.append(download_process1)
    # all_process.append(download_process2)
    #
    # # recognition only one process
    # recog_process = multiprocessing.Process(target=seg_model.recognition)
    # all_process.append(recog_process)
    #
    # save_process1 = multiprocessing.Process(target=seg_model.save, args=(args.dir, ))
    # save_process2 = multiprocessing.Process(target=seg_model.save, args=(args.dir,))
    # all_process.append(save_process1)
    # all_process.append(save_process2)
    #
    # for proc_ in all_process:
    #     if not isinstance(proc_, multiprocessing.Process):
    #         break
    #     proc_.start()
    #
    # for proc_ in all_process:
    #     if not isinstance(proc_, multiprocessing.Process):
    #         break
    #     proc_.join()

    time2 = time.time()

    print("finish in {} s\n".format(time2 - time1))

