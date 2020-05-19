import argparse
import numpy as np
import os

from distutils.util import strtobool
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LassoCV, SGDClassifier, SGDRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import matthews_corrcoef, make_scorer
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
                    "matthews_corrcoef": matthews_corrcoef}


def main(args):
    dataset_arrays = [
        ml_utils.create_samples(args.dataset_names[dataset], args.dataset_paths[dataset], args.mat_files[dataset],
                                args.balanced[dataset], bool(strtobool(args.save_images))) for dataset in
        range(len(args.dataset_names))]
    total_images = len(dataset_arrays[0][4][0])
    for dataset in range(1, len(dataset_arrays)):
        dataset_arrays[dataset][3][:, 0] += total_images
        total_images += len(dataset_arrays[dataset][4][0])
    selected_indices, selected_features = ml_utils.get_feature_subset(args, dataset_arrays[0][2])
    log_string = "Selected features:\n%s\n" % selected_features
    print(log_string)

    parameter_dict = ml_utils.get_parameter_dict(args)
    clf = available_models[args.model](**parameter_dict)
    scoring = {}
    for metric in args.metrics:
        score_func = available_scores[metric]
        scoring[metric] = make_scorer(score_func) if score_func is not None else None
    scorer_names = list(scoring.keys())

    images = [i for i in range(dataset_arrays[-1][3][-1, 0])]
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=0)
    kf.get_n_splits(images)
    folds = []
    test_images = []
    prev_pixels = [0]
    for dataset in range(1, len(dataset_arrays)):
        prev_pixels += [dataset_arrays[dataset - 1][1].shape[0] + prev_pixels[dataset - 1]]
    all_paths = [dataset_arrays[dataset][4][0][path] for dataset in range(len(dataset_arrays)) for path in
                 range(len(dataset_arrays[dataset][4][0]))]
    for train_index, test_index in kf.split(images):
        folds.append((
            np.concatenate(
                [np.uint32(prev_pixels[dataset] + np.where(dataset_arrays[dataset][3][:, 0] == idx)[0])
                 for idx in train_index
                 for dataset in range(len(dataset_arrays))]
            ),
            np.concatenate(
                [np.uint32(prev_pixels[dataset] + np.where(dataset_arrays[dataset][3][:, 0] == idx)[0])
                 for idx in test_index
                 for dataset in range(len(dataset_arrays))]
            )
        ))
        fold_test_images = ["" for i in range(len(test_index))]
        for image, idx in enumerate(test_index):
            fold_test_images[image] = os.path.split(all_paths[idx])[-1]
        test_images.append(sorted(fold_test_images))
    del kf
    test_images = "\n".join([str(fold) for fold in test_images])

    paths = np.concatenate([dataset_arrays[dataset][4] for dataset in range(len(dataset_arrays))], axis=1).transpose()
    x = np.concatenate([dataset_arrays[dataset][0] for dataset in range(len(dataset_arrays))], axis=0)
    y = np.concatenate([dataset_arrays[dataset][1] for dataset in range(len(dataset_arrays))], axis=0)
    feature_names = dataset_arrays[0][2]
    selected_pixels = np.concatenate([dataset_arrays[dataset][3] for dataset in range(len(dataset_arrays))], axis=0)
    del dataset_arrays

    if not os.path.exists(args.save_results_to):
        os.makedirs(args.save_results_to)
    with open(os.path.join(args.save_results_to, "results.txt"), "w") as f:
        f.write(log_string)
        print("Classifier: %s" % str(clf))
        f.write("\nClassifier: %s\n" % str(clf))
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
    parser.add_argument("--dataset_names", type=str, nargs="+",
                        help="Must be one of: 'cfd', 'cfd-pruned', 'aigle-rn', 'esar'")
    parser.add_argument("--dataset_paths", type=str, nargs="+",
                        help="Path to the folder containing the dataset as downloaded from the original source")
    parser.add_argument("--mat_files", type=str, nargs="+", default=None,
                        help="Path to a mat file containing the processed "
                             "dataset. If None is provided, the dataset will be "
                             "processed")
    parser.add_argument("--balanced", type=int, nargs="+", default=[0],
                        help="0: save all the resulting pixels; 1: randomly sample background pixels equal to the "
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
    return args_dict


if __name__ == "__main__":
    args = parse_args()
    main(args)
