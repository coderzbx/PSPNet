import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=False)
    parser.add_argument('--text_dir', type=str, required=False)
    args = parser.parse_args()

    image_dir = args.image_dir
    txt_dir = args.text_dir

    image_list = os.listdir(image_dir)
    with open(txt_dir, "wb") as f:
        for id in image_list:
            name_list = id.split(".")
            name_ext = name_list[len(name_list) - 1]
            if name_ext != 'png' and name_ext != 'jpg':
                continue

            str = id + '\t' + id + '\n'
            f.write(str)