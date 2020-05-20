import cv2
import data
import numpy as np
import os
import scipy.io

from sklearn.metrics import accuracy_score, r2_score, make_scorer
from sklearn.model_selection import cross_validate, KFold


# def create_images(dataset_name, dataset_path, mat_path=None):
#     print("Loading images...")
#     if dataset_name == "cfd" or dataset_name == "cfd-pruned":
#         or_im_paths, gt_paths = data.paths_generator_cfd(dataset_path)
#     elif dataset_name == "aigle-rn":
#         or_im_paths, gt_paths = data.paths_generator_crack_dataset(dataset_path, "AIGLE_RN")
#     elif dataset_name == "esar":
#         or_im_paths, gt_paths = data.paths_generator_crack_dataset(dataset_path, "ESAR")
#     ground_truth, gt_paths = data.images_from_paths(gt_paths)
#     ground_truth = np.array(ground_truth, dtype=np.float32) / 255
#     if mat_path is None:
#         images, feature_names, file_names = get_morphological_features(or_im_paths, dataset_name)
#     else:
#         images, feature_names, file_names = open_morphological_features(mat_path)
#     print("Images loaded!")
#     return images, ground_truth, feature_names, file_names


# def get_morphological_features(paths, dataset_name):
#     paths = ";".join(paths)
#     command = "matlab -nodesktop -nojvm -r 'try preprocess_images(\"%s\",\"%s\"); catch; end; quit'" % (
#         paths, dataset_name)
#     os.system(command)
#     images, feature_names, file_names, or_im_shape = open_morphological_features(dataset_name + ".mat")
#     return images, feature_names, file_names
#
#
# def open_morphological_features(path_to_mat):
#     mat_root = os.path.split(path_to_mat)[0]
#
#     mat_files = sorted([f for f in os.listdir(path_to_mat)
#                         if not f.startswith(".") and f.endswith(".mat")],
#                        key=lambda f: f.lower())
#     images = np.array([scipy.io.loadmat(os.path.join(path_to_mat, mat_file))["images"] for mat_file in mat_files],
#                       dtype=np.float32)
#
#     try:
#         feature_names = scipy.io.loadmat(os.path.join(mat_root, "feature_names.mat"))["feature_names"]
#         for feature in range(len(feature_names)):
#             feature_names[feature] = feature_names[feature].strip()
#     except FileNotFoundError:
#         print("No mat file found for feature names.")
#         feature_names = None
#
#     return images, feature_names, mat_files


def get_feature_subset(args, feature_names):
    if args.features_subset == "all":
        selected_features = ";".join(feature_names)
    else:
        with open(args.features_subset, "r") as feat_file:
            feat_subset = feat_file.read().strip()
            if feat_subset == "all":
                selected_features = ";".join(feature_names)
            else:
                selected_features = feat_subset.replace("*", "").replace("\n", ";")

    selected_indices = [np.where(feature_names == feature)[0][0] for feature in selected_features.split(";")]
    selected_features = "*%s" % selected_features.replace(";", "\n*")
    return selected_indices, selected_features


def get_parameter_dict(args):
    parameter_dict = {}
    for idx in range(int(len(args.model_parameters) / 2)):
        parameter_name = args.model_parameters[2 * idx]
        parameter = args.model_parameters[2 * idx + 1]
        if parameter == "None":
            parameter_dict[parameter_name] = None
            continue
        try:
            if "." in parameter:
                parameter_dict[parameter_name] = float(parameter)
                continue
            else:
                parameter_dict[parameter_name] = int(parameter)
                continue
        except ValueError:
            if parameter.startswith("[") and parameter.endswith("]"):
                parameter = parameter[1:-1].split(",")
                for idx, element in enumerate(parameter):
                    try:
                        if "." in element:
                            parameter[idx] = float(element)
                        else:
                            parameter[idx] = int(element)
                    except ValueError:
                        parameter[idx] = element
                parameter_dict[parameter_name] = parameter
            elif parameter.startswith("{") and parameter.endswith("}"):
                parameter = parameter[1:-1].split(",")
                sub_dict = {}
                for idx, element in enumerate(parameter):
                    pair = element.split(":")
                    try:
                        if "." in pair[0]:
                            key = float(pair[0])
                        else:
                            key = int(pair[0])
                    except ValueError:
                        key = pair[0]
                    try:
                        if "." in pair[1]:
                            sub_dict[key] = float(pair[1])
                        else:
                            sub_dict[key] = int(pair[1])
                    except ValueError:
                        sub_dict[key] = pair[1]
                parameter_dict[parameter_name] = sub_dict
            else:
                parameter_dict[parameter_name] = parameter
    return parameter_dict


def create_samples(dataset_name, dataset_path, mat_path=None, balanced=False, save_images=True):
    print("Loading data...")

    if dataset_name == "cfd" or dataset_name == "cfd-pruned":
        or_im_paths, gt_paths = data.paths_generator_cfd(dataset_path)
    elif dataset_name == "aigle-rn":
        or_im_paths, gt_paths = data.paths_generator_crack_dataset(dataset_path, "AIGLE_RN")
    elif dataset_name == "esar":
        or_im_paths, gt_paths = data.paths_generator_crack_dataset(dataset_path, "ESAR")

    if mat_path is not None:
        features, labels, feature_names, selected_pixels = open_morphological_features(mat_path, balanced)
        print("Data loaded!")
        return features, labels, feature_names, selected_pixels, [or_im_paths, gt_paths]

    features, labels, feature_names, selected_pixels = get_morphological_features(or_im_paths, gt_paths, dataset_name,
                                                                                  balanced, save_images)
    print("Data loaded!")
    return features, labels, feature_names, selected_pixels, [or_im_paths, gt_paths]


def create_multidataset_samples(dataset_names, dataset_paths, mat_paths=[None], balanceds=[False], save_images=True):
    print("Loading data...")
    if len(mat_paths) == 1:
        mat_paths = [mat_paths[0] for i in range(len(dataset_names))]
    if len(balanceds) == 1:
        balanceds = [balanceds[0] for i in range(len(dataset_names))]

    features_list, labels_list, feature_names_list, selected_pixels_list, paths_list = [], [], [], [], []
    acumulated_images = 0
    for idx in range(len(dataset_names)):
        dataset_name, dataset_path, mat_path, balanced, save_images = dataset_names[idx], dataset_paths[idx], mat_paths[
            idx], balanceds[idx], save_images

        if dataset_name == "cfd" or dataset_name == "cfd-pruned":
            or_im_paths, gt_paths = data.paths_generator_cfd(dataset_path)
        elif dataset_name == "aigle-rn":
            or_im_paths, gt_paths = data.paths_generator_crack_dataset(dataset_path, "AIGLE_RN")
        elif dataset_name == "esar":
            or_im_paths, gt_paths = data.paths_generator_crack_dataset(dataset_path, "ESAR")

        if mat_path is not None:
            features, labels, feature_names, selected_pixels = open_morphological_features(mat_path, balanced)
            selected_pixels[:, 0] = acumulated_images + selected_pixels[:, 0]
            acumulated_images += len(set(selected_pixels[:, 0]))
            features_list.append(features)
            labels_list.append(labels)
            feature_names_list.append(feature_names)
            selected_pixels_list.append(selected_pixels)
            paths_list.append([or_im_paths, gt_paths])
            continue

        features, labels, feature_names, selected_pixels = get_morphological_features(or_im_paths, gt_paths,
                                                                                      dataset_name, balanced,
                                                                                      save_images)
        selected_pixels[:, 0] = acumulated_images + selected_pixels[:, 0]
        acumulated_images += len(selected_pixels[:, 0])
        features_list.append(features)
        labels_list.append(labels)
        feature_names_list.append(feature_names)
        selected_pixels_list.append(selected_pixels)
        paths_list.append([or_im_paths, gt_paths])

    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    feature_names = feature_names_list[0]
    selected_pixels = np.concatenate(selected_pixels_list, axis=0)
    paths = np.concatenate(paths_list, axis=1)
    print("Data loaded!")
    return features, labels, feature_names, selected_pixels, paths


def get_morphological_features(paths, gt_paths, dataset_name, balanced, save_resulting_images):
    paths = ";".join(paths)
    gt_paths = ";".join(gt_paths)
    command = "matlab -nodesktop -nojvm -r 'try preprocess_images(\"%s\",\"%s\",\"%s\",%s,%s); catch; end; quit'" % (
        paths, gt_paths, dataset_name, str(balanced).lower(), str(save_resulting_images).lower())
    os.system(command)
    if balanced is True:
        balanced_string = "_balanced"
    elif balanced > 0:
        balanced_string = "_1_to_%s" % str(balanced).replace(".", ",")
    elif balanced < 0:
        balanced_string = "_1_to_%s_weighted" % str(-balanced).replace(".", ",")
    else:
        balanced_string = ""
    features, labels, feature_names, selected_pixels = open_morphological_features(
        dataset_name + balanced_string + ".mat", balanced)
    return features, labels, feature_names, selected_pixels


def open_morphological_features(path_to_mat, balanced=False):
    mat_root, dataset_name = os.path.split(path_to_mat)

    if balanced is True:
        balanced_string = "_balanced"
    elif balanced > 0:
        balanced_string = "_1_to_%s" % str(balanced).replace(".", ",")
    elif balanced < 0:
        balanced_string = "_1_to_%s_weighted" % str(-balanced).replace(".", ",")
    else:
        balanced_string = ""

    dataset_name = dataset_name.split(balanced_string + ".mat")[0]
    features = scipy.io.loadmat(path_to_mat)["data"]
    labels = scipy.io.loadmat(os.path.join(mat_root, dataset_name + balanced_string + "_labels.mat"))["labels"]

    try:
        feature_names = scipy.io.loadmat(os.path.join(mat_root, dataset_name + balanced_string + "_feature_names.mat"))[
            "feature_names"]
        for feature in range(len(feature_names)):
            feature_names[feature] = feature_names[feature].strip()
    except FileNotFoundError:
        print("No mat file found for feature names.")
        feature_names = None

    try:
        selected_pixels = scipy.io.loadmat(os.path.join(mat_root, dataset_name + balanced_string + "_pick_maps.mat"))[
            "pick_maps"]
    except FileNotFoundError:
        print("No mat file found for picked pixels.")
        selected_pixels = None

    return features, np.ravel(labels), feature_names, selected_pixels.astype(np.uint16)


def flatten_pixels(images_array):
    print("Flattening images.")
    array_shape = images_array.shape

    if len(array_shape) == 3:
        return np.reshape(images_array, (array_shape[0] * array_shape[1] * array_shape[2],), "F"), array_shape
    elif len(array_shape) == 4:
        return np.reshape(images_array, (array_shape[0] * array_shape[1] * array_shape[2], array_shape[3]),
                          "F"), array_shape


def reconstruct_from_flat_pixels(flatten_pixels_array, original_shape):
    print("Reconstructing images.")
    flatten_shape = flatten_pixels_array.shape
    return np.reshape(flatten_pixels_array, original_shape, "F"), flatten_shape


def reconstruct_from_selected_pixels(selected_pixels, predicted_labels, real_labels, paths):
    or_paths = paths[0]
    gt_paths = paths[1]
    current_image = 0
    gt = cv2.imread(gt_paths[0])
    n_white = np.sum(gt) / 255
    n_black = gt.shape[0] * gt.shape[1] - n_white
    if n_black < n_white:
        gt = 255 - gt
    img = cv2.imread(or_paths[0])
    predicted_image = np.zeros(gt.shape, dtype=np.uint8)
    predicted_cracks = np.zeros(gt.shape, dtype=np.uint8)
    resulting_images = []

    for pixel in range(len(selected_pixels)):
        image, row, col = selected_pixels[pixel]
        if image > current_image:
            resulting_images.append(np.concatenate((img, gt, predicted_cracks, predicted_image), axis=1))
            gt = cv2.imread(gt_paths[image])
            n_white = np.sum(gt) / 255
            n_black = gt.shape[0] * gt.shape[1] - n_white
            if n_black < n_white:
                gt = 255 - gt
            img = cv2.imread(or_paths[image])
            predicted_image = np.zeros(gt.shape, dtype=np.uint8)
            predicted_cracks = np.zeros(gt.shape, dtype=np.uint8)
            current_image = image
        if predicted_labels[pixel] == 1 or predicted_labels[pixel] == 0:
            if predicted_labels[pixel] == real_labels[pixel]:
                if real_labels[pixel] == 1:
                    predicted_image[row, col, :] = np.array([0, 255, 0], dtype=np.uint8)
                else:
                    predicted_image[row, col, :] = np.array([255, 0, 0], dtype=np.uint8)
            else:
                predicted_image[row, col, :] = np.array([0, 0, 255], dtype=np.uint8)
        else:
            regression_value = max(0, predicted_labels[pixel])
            regression_value = min(1, regression_value)
            regression_value = 255 * regression_value
            predicted_image[row, col, :] = np.array([regression_value, regression_value, regression_value],
                                                    dtype=np.uint8)
        predicted_cracks[row, col, :] = np.array([255 * predicted_labels[pixel] for i in range(3)])
    resulting_images.append(np.concatenate((img, gt, predicted_cracks, predicted_image), axis=1))
    return resulting_images


def save_visual_results(selected_pixels, predicted_labels, real_labels, paths, path_dir="resulting_images"):
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

    results = reconstruct_from_selected_pixels(selected_pixels, predicted_labels, real_labels, paths)
    for idx, image_path in enumerate(paths[0]):
        file_name = os.path.split(image_path)[1]
        cv2.imwrite(os.path.join(path_dir, file_name), results[idx])


def cross_validate_predict(data, folds, cv_results):
    n_samples, n_features = data.shape
    predicts = np.zeros((n_samples,))
    for idx, fold in enumerate(folds):
        indices = fold[1]
        estimator = cv_results["estimator"][idx]
        fold_results = estimator.predict(data[indices])
        for fold_idx, result in enumerate(fold_results):
            predicts[indices[fold_idx]] = result
    return predicts


def cross_dataset_validation(model, x_train, y_train, x_test, y_test, test_selected_pixels, test_paths,
                             save_images_to=None, score_function=None):
    model.fit(x_train, y_train)
    cross_dataset_predictions = model.predict(x_test)
    if score_function is None:
        if set(cross_dataset_predictions) == {0, 1}:
            score_function = accuracy_score
        else:
            score_function = r2_score
    cross_dataset_score = score_function(y_test, cross_dataset_predictions)

    if save_images_to is not None:
        save_visual_results(test_selected_pixels, cross_dataset_predictions, y_test, test_paths, save_images_to)
    return cross_dataset_score, score_function.__name__


def n_fold_cross_validation(args, available_models, available_scores, x, y, feature_names, selected_pixels, paths):
    selected_indices, selected_features = get_feature_subset(args, feature_names)
    log_string = "Selected features:\n%s\n" % selected_features
    print(log_string)

    parameter_dict = get_parameter_dict(args)
    clf = available_models[args.model](**parameter_dict)
    scoring = {}
    for metric in args.metrics:
        score_func = available_scores[metric]
        scoring[metric] = make_scorer(score_func) if score_func is not None else None
    scorer_names = list(scoring.keys())
    if "dsc" in scorer_names:
        del scoring["dsc"]

    images = list(set(selected_pixels[:, 0]))
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=0)
    kf.get_n_splits(y)
    folds = []
    test_images = []
    for train_index, test_index in kf.split(images):
        folds.append((np.concatenate([np.uint32(np.where(selected_pixels[:, 0] == idx)[0]) for idx in train_index]),
                      np.concatenate([np.uint32(np.where(selected_pixels[:, 0] == idx)[0]) for idx in test_index])))
        fold_test_images = ["" for i in range(len(test_index))]
        for image, idx in enumerate(test_index):
            fold_test_images[image] = os.path.split(paths[0][idx])[-1]
        test_images.append(sorted(fold_test_images))
    del kf
    test_images = "\n".join([str(fold) for fold in test_images])

    if not os.path.exists(args.save_results_to):
        os.makedirs(args.save_results_to)
    with open(os.path.join(args.save_results_to, "results.txt"), "w") as f:
        f.write(log_string)
        print("Classifier: %s" % str(clf))
        f.write("\nClassifier: %s\n" % str(clf))
        cv_results = cross_validate(clf, x[:, selected_indices], y, scoring=scoring, verbose=50, n_jobs=1,
                                    cv=folds, return_estimator=True)
        dsc = False
        for scorer_name in scorer_names:
            if scorer_name == "dsc":
                dsc = True
                continue
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

        predictions = cross_validate_predict(x[:, selected_indices], folds, cv_results)
        save_visual_results(selected_pixels, predictions, y, paths, args.save_results_to)
    if dsc:
        calculate_dsc_from_result_folder(args.save_results_to)


def get_image_dsc(img):
    if type(img) is str:
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    elif type(img) is not np.ndarray:
        raise ValueError("Expected input: either a path to an image, or an image as numpy array.")
    height, width = img.shape
    width = int(width / 4)

    gt = img[:, width:2 * width]
    ret, gt = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)
    gt = (gt / 255).astype(np.uint8)
    n_white = np.sum(gt)
    n_black = gt.shape[0] * gt.shape[1] - n_white
    if n_black < n_white:
        gt = 1 - gt

    predicted = img[:, 2 * width:3 * width]
    ret, predicted = cv2.threshold(predicted, 127, 255, cv2.THRESH_BINARY)
    predicted = (predicted / 255).astype(np.uint8)

    intersect = predicted * gt
    intersect_area = np.sum(intersect)
    gt_area = np.sum(gt)
    predicted_crack_area = np.sum(predicted)
    dsc = (2 * intersect_area + 1) / (gt_area + predicted_crack_area + 1)
    # cv2.imshow("%s, %s, %s, %s" % (gt_area, predicted_crack_area, intersect_area, dsc), np.concatenate((gt, predicted, intersect, gt * (1 - predicted)), axis=1) * 255)
    # cv2.waitKey(1000)
    # cv2.destroyWindow("%s, %s, %s, %s" % (gt_area, predicted_crack_area, intersect_area, dsc))
    return np.float32(dsc)


def calculate_dsc_from_result_folder(result_folder):
    with open(os.path.join(result_folder, "results.txt"), 'r') as f:
        content = f.read().strip().split("\n")
    folds_first_line = None
    for line_number, line in enumerate(content):
        if line.startswith("Test images per fold"):
            folds_first_line = line_number + 1
            break
    if folds_first_line is None:
        raise ValueError("results.txt doesn't have the expected format. Try a different file.")
    fold_scores = [get_set_dsc(result_folder, line.strip('][').replace("'", '').split(', ')) for line in
                   content[folds_first_line:]]
    scores = [np.mean(fold) for fold in fold_scores]
    statistics = "{}(%) Avg,Std,Min,Max = {:.2f},{:.2f},{:.2f},{:.2f}".format("DSC", 100 * np.mean(scores),
                                                                              100 * np.std(scores),
                                                                              100 * np.min(scores),
                                                                              100 * np.max(scores))
    scores = "%s\n %s" % (str(scores[:6]).strip("]").replace(",", ""), str(scores[6:]).strip("[").replace(",", ""))
    print("\n%s" % scores)
    print(statistics)
    new_text = content[:folds_first_line - 1] + [scores, "%s\n" % statistics] + content[folds_first_line - 1:]
    new_text = "\n".join(new_text)
    with open(os.path.join(result_folder, "results.txt"), 'w') as f:
        f.write(new_text.strip())


def get_set_dsc(images_root_path, images_list):
    if images_root_path is None:
        scores = [get_image_dsc(image) for image in images_list]
    else:
        scores = [get_image_dsc(os.path.join(images_root_path, image)) for image in images_list]
    return np.array(scores)

# calculate_dsc_from_result_folder("/media/winbuntu/google-drive/ESIEE/Thesis/Year_1/results_crack_detection/Machine Learning/tree_based_good/cfd-pruned feature_set_3 RandomForestClassifier 50")
