import argparse
import cv2
import numpy as np
import smilPython as sp


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str)
    parser.add_argument("--path_sizes", nargs="+", type=int, default=[10], required=True)
    args_dict = parser.parse_args(args)
    return args_dict


def main(args):
    image = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
    image = (image.astype(np.float) - image.min()) * 255 / (image.max() - image.min())
    image = image.astype(np.uint8)
    original = sp.Image()
    original.fromNumArray(image.transpose())

    closed = sp.Image()
    for size in args.path_sizes:
        sp.ImPathClosing(original, size, closed)
        sp.write(closed, "matlab_closing_%s.png" % size)


if __name__ == "__main__":
    args = parse_args()
    main(args)
