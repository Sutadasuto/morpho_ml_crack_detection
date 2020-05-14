import argparse

from distutils.util import strtobool

import ml_utils


def main(args):
    x, y, feature_names, selected_pixels, paths = ml_utils.create_samples(args.dataset_name, args.dataset_path, None,
                                                                          args.balanced,
                                                                          bool(strtobool(args.save_images)))
    return x, y, feature_names, selected_pixels, paths


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str, required=True,
                        help="Must be one of: 'cfd', 'cfd-pruned', 'aigle-rn', 'esar'")
    parser.add_argument("dataset_path", type=str, required=True,
                        help="Path to the folder containing the dataset as downloaded from the original source")
    parser.add_argument("--balanced", type=int, default=0,
                        help="0: save all the resulting pixels; 1: randomly sample background pixels equal to the "
                             "number of crack pixels per image, and save crack and non-crack sampled pixels; N: "
                             "randomly sample N background pixels for each crack pixel per image, and save crack and "
                             "non-crack sampled pixels; -N: randomly sample weighted background pixels equal to the "
                             "number of crack pixels per image, and save crack and non-crack sampled pixels (weighting "
                             "is done according to the intensity resulting from applying a Frangi filter)")
    parser.add_argument("--save_images", type=str, default="True",
                        help="'True' or 'False': saving the extracted features in the same location that the folder "
                             "containing the images in the dataset")
    args_dict = parser.parse_args(args)
    return args_dict


if __name__ == "__main__":
    args = parse_args()
    main(args)
