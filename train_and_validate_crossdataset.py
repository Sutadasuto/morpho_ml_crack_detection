import argparse
import numpy as np
import os

from distutils.util import strtobool
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LassoCV, SGDClassifier, SGDRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import cross_validate, KFold

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
                    "recall_score": precision_score,
                    "f1_score": f1_score}


def main(args):
    x_train, y_train, feature_names, selected_pixels_train, paths_train = ml_utils.create_samples(args.dataset_names[0],
                                                                                                  args.dataset_paths[0],
                                                                                                  args.mat_files[0],
                                                                                                  args.balanced[0],
                                                                                                  bool(strtobool(
                                                                                                      args.save_images))
                                                                                                  )
    x_test, y_test, feature_names, selected_pixels_test, paths_test = ml_utils.create_samples(args.dataset_names[1],
                                                                                                  args.dataset_paths[1],
                                                                                                  args.mat_files[1],
                                                                                                  args.balanced[1],
                                                                                                  bool(strtobool(
                                                                                                      args.save_images))
                                                                                                  )
    selected_indices, selected_features = ml_utils.get_feature_subset(args, feature_names)
    log_string = "Selected features:\n%s\n" % selected_features
    print(log_string)

    parameter_dict = ml_utils.get_parameter_dict(args)
    clf = available_models[args.model](**parameter_dict)
    scoring = {}
    for metric in args.metrics:
        score_func = available_scores[metric]
        scoring[metric] = make_scorer(score_func) if score_func is not None else None
    scorer_names = list(scoring.keys())

    # images = list(set(selected_pixels[:, 0]))
    # kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=0)
    # kf.get_n_splits(y)
    # folds = []
    # test_images = []
    # for train_index, test_index in kf.split(images):
    #     folds.append((np.concatenate([np.uint32(np.where(selected_pixels[:, 0] == idx)[0]) for idx in train_index]),
    #                   np.concatenate([np.uint32(np.where(selected_pixels[:, 0] == idx)[0]) for idx in test_index])))
    #     fold_test_images = ["" for i in range(len(test_index))]
    #     for image, idx in enumerate(test_index):
    #         fold_test_images[image] = os.path.split(paths[0][idx])[-1]
    #     test_images.append(sorted(fold_test_images))
    # del kf
    # test_images = "\n".join([str(fold) for fold in test_images])

    if not os.path.exists(args.save_results_to):
        os.makedirs(args.save_results_to)
    with open(os.path.join(args.save_results_to, "results.txt"), "w") as f:
        f.write(log_string)
        print("Classifier: %s" % str(clf))
        f.write("\nClassifier: %s\n" % str(clf))
        clf.fit(x_train, y_train)

        cv_results = cross_validate(clf, x[:, selected_indices], y, scoring=scoring, verbose=50, n_jobs=1,
                                    cv=folds, return_estimator=True)
        for scorer_name in scorer_names:
            scores = cv_results["test_" + scorer_name]
            print("\n%s" % str(scores))
            f.write("\n%s\n" % str(scores))
            print("{}(%) Avg,Std,Min,Max = {:.2f},{:.2f},{:.2f},{:.2f}".format(scorer_name, 100 * np.mean(scores),
                                                                               100 * np.std(scores),
                                                                               100 * np.min(scores),
                                                                               100 * np.max(scores)))
            f.write("{}(%) Avg,Std,Min,Max = {:.2f},{:.2f},{:.2f},{:.2f}\n".format(scorer_name, 100 * np.mean(scores),
                                                                                   100 * np.std(scores),
                                                                                   100 * np.min(scores),
                                                                                   100 * np.max(scores)))
        print("\nTest images per fold (dataset %s):\n%s" % (args.dataset_name, test_images))
        f.write("\nTest images per fold (dataset %s):\n%s" % (args.dataset_name, test_images))

        predictions = ml_utils.cross_validate_predict(x[:, selected_indices], folds, cv_results)
        ml_utils.save_visual_results(selected_pixels, predictions, y, paths, args.save_results_to)
    ml_utils.calculate_dsc_from_result_folder(args.save_results_to)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_names", type=str, nargs=2,
                        help="Must be one of: 'cfd', 'cfd-pruned', 'aigle-rn', 'esar'")
    parser.add_argument("--dataset_paths", type=str, nargs=2,
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
    if args_dict.balanced == 0 or args_dict.balanced == 1:
        args_dict.balanced = bool(args_dict.balanced)
    if len(args_dict.mat_files) == 1:
        args_dict.mat_files = [args_dict.mat_files[0] for i in range(len(args_dict.dataset_names))]
    return args_dict


if __name__ == "__main__":
    args = parse_args()
    main(args)