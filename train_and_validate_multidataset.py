import argparse

from distutils.util import strtobool
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LassoCV, SGDClassifier, SGDRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, f1_score

import ml_utils

available_models = {"RandomForestClassifier": RandomForestClassifier,
                    "RandomForestRegressor": RandomForestRegressor,
                    "Lasso": Lasso,
                    "LassoCV": LassoCV,
                    "SGDClassifier": SGDClassifier,
                    "SGDRegressor": SGDRegressor,
                    "DecisionTreeClassifier": DecisionTreeClassifier,
                    "DecisionTreeRegressor": DecisionTreeRegressor}

available_scores = {"default": None,
                    "matthews_corrcoef": matthews_corrcoef,
                    "precision_score": precision_score,
                    "recall_score": recall_score,
                    "f1_score": f1_score,
                    "dsc": "dsc"}


def main(args):
    x, y, feature_names, selected_pixels, paths = ml_utils.create_multidataset_samples(args.dataset_names,
                                                                                       args.dataset_paths, args.mat_files,
                                                                                       args.balanced,
                                                                                       bool(strtobool(args.save_images))
                                                                                       )
    args.dataset_name = "/".join(args.dataset_names)
    ml_utils.n_fold_cross_validation(args, available_models, available_scores, x, y, feature_names, selected_pixels,
                                     paths)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_names", type=str, nargs="+",
                        help="Must be one of: 'cfd', 'cfd-pruned', 'aigle-rn', 'esar'")
    parser.add_argument("--dataset_paths", type=str, nargs="+",
                        help="Path to the folder containing the datasets as downloaded from the original source")
    parser.add_argument("--mat_files", type=str, nargs="+", default=[None],
                        help="Path to mat files containing the processed "
                             "datasets. If None is provided, the dataset will be "
                             "processed")
    parser.add_argument("--balanced", type=int, nargs="+", default=[0],
                        help="Either a list of integers or ar integer to be shared by all datasets. 0: save all the "
                             "resulting pixels; 1: randomly sample background pixels equal to the "
                             "number of crack pixels per image, and save crack and non-crack sampled pixels; N: "
                             "randomly sample N background pixels for each crack pixel per image, and save crack and "
                             "non-crack sampled pixels; -N: randomly sample weighted background pixels equal to the "
                             "number of crack pixels per image, and save crack and non-crack sampled pixels (weighting "
                             "is done according to the intensity resulting from applying a Frangi filter)")
    parser.add_argument("--save_images", type=str, default="True",
                        help="'True' or 'False': saving the extracted features in the same location that the folder "
                             "containing the images in the dataset")
    parser.add_argument("--model", type=str, default="DecisionTreeClassifier",
                        help="Model used for training. Available models "
                             "are %s" % ", ".join(available_models.keys()))
    parser.add_argument("--model_parameters", type=str, nargs="*", default=['class_weight', 'balanced', 'random_state',
                                                                            '0'],
                        help="Parameters for the ML model provided as a "
                             "list: 'Parameter_1' 'Value_1' 'Parameter_2' "
                             "'Value_2' ...")
    parser.add_argument("--save_results_to", type=str, default="results",
                        help="Predicted images and evaluation scores will be "
                             "saved into this folder")
    parser.add_argument("--n_folds", type=int, default=10, help="Number of folds for n-fold cross validation")
    parser.add_argument("--metrics", type=str, nargs="+", default=["matthews_corrcoef"],
                        help="List of metrics used for validation. Available metrics are %s" % ", ".join(
                            available_scores.keys()))
    parser.add_argument("--features_subset", type=str, default="all",
                        help="If training with just a subset of features, this parameter should be a path to a .txt "
                             "file containing a list of N features formatted like this: "
                             "'*Feature_1\n*Feature_2\n...*Feature_N'. "
                             "File can contain only 'all', to choose all features")

    args_dict = parser.parse_args(args)
    if len(args_dict.balanced) == 1:
        args_dict.balanced = [args_dict.balanced[0] for i in range(len(args_dict.dataset_names))]
    for dataset in range(len(args_dict.balanced)):
        if args_dict.balanced[dataset] == 0 or args_dict.balanced[dataset] == 1:
            args_dict.balanced[dataset] = bool(args_dict.balanced[dataset])
    if len(args_dict.mat_files) == 1:
        args_dict.mat_files = [args_dict.mat_files[0] for i in range(len(args_dict.dataset_names))]
    return args_dict


if __name__ == "__main__":
    args = parse_args()
    main(args)
