import os
import cv2
import argparse


class FormatTrainSet:
    def __init__(self):
        return

    def format(self, image_dir, label_dir, txt_dir, image_type):
        image_files = os.listdir(image_dir)

        images = []
        annots = []

        for id_ in image_files:
            file_name = id_.split('.')
            file_ex = file_name[1]
            if file_ex != 'png' and file_ex != 'jpg':
                continue
            file_name = file_name[0]
            file_name_list = file_name.split("_")
            file_id = ''
            file_name_list = file_name_list[:-1]
            for name in file_name_list:
                file_id += name + "_"

            image_name = file_id + "leftImg8bit.png"
            label_name = file_id + "gtFine_labelIds.png"

            images.append(os.path.join(image_dir, image_name))
            annots.append(os.path.join(label_dir, label_name))

        images.sort()
        annots.sort()
        image_count = len(images)
        label_count = len(annots)

        train_txt = os.path.join(txt_dir, '{}.txt'.format(image_type))
        with open(train_txt, 'wb') as f:
            if image_count == label_count:
                for image, annot in zip(images, annots):
                    str = image + ' ' + annot + '\n'
                    f.write(str.encode("UTF-8"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=False)
    parser.add_argument('--annot_dir', type=str, required=False)
    parser.add_argument('--text_dir', type=str, required=False)
    args = parser.parse_args()


    handle = FormatTrainSet()
    image_dir = args.image_dir
    annot_dir = args.annot_dir
    txt_dir = args.text_dir
    image_type = 'train'
    handle.format(image_dir=image_dir, label_dir=annot_dir, txt_dir=txt_dir, image_type=image_type)


