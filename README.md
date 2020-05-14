# morpho_ml_crack_detection
Clean version of the ML approach first introduced in https://github.com/Sutadasuto/morpho_crack_detection/tree/petr_advice

To validate a model using fold cross-vlidation, run:
```
python train_and_validate.py dataset_name path_to_dataset_folder
```
If you have already extracted features using this repository, you can add:
```
--mat_file path_to_mat
```
to train straightforward without extracting features again.

Mat file name should be "dataset_name.mat". This repository creates 4 mat files when extracting features; the 4 files should always remain together in the same directory and their names shouldn't be changed (as long as they share directory and their names are unchanged, they can be moved to any desired location).

## Pre-requisites
Feature extraction needs Matlab and SMIL (http://smil.cmm.mines-paristech.fr/wiki/doku.php/start) compiled from source with the Path Opening Addon.
Additional Python requirements can be met by creating a conda environment with the environment.yaml file provided in this repository.

The full list of parameters is shown next:

* ("dataset_name", type=str, help="Must be one of: 'cfd', 'cfd-pruned', 'aigle-rn', 'esar'")
* ("dataset_path", type=str, help="Path to the folder containing the dataset as downloaded from the original source")
* ("--mat_file", type=str, default=None, help="Path to a mat file containing the processed dataset. If None is provided, the dataset will be processed")
* ("--balanced", type=int, default=0, help="0: save all the resulting pixels; 1: randomly sample background pixels equal to the number of crack pixels per image, and save crack and non-crack sampled pixels; N: randomly sample N background pixels for each crack pixel per image, and save crack and non-crack sampled pixels; -N: randomly sample weighted background pixels equal to the number of crack pixels per image, and save crack and non-crack sampled pixels (weighting is done according to the intensity resulting from applying a Frangi filter)")
* ("--save_images", type=str, default="True", help="'True' or 'False': saving the extracted features in the same location that the folder containing the images in the dataset")
* ("--model", type=str, default="DecisionTreeClassifier", help="Model used for training. Available models are %s" % ", ".join(available_models.keys()))
* ("--model_parameters", type=str, nargs="*", default=['class_weight', 'balanced', 'random_state', '0'], help="Parameters for the ML model provided as a list: 'Parameter_1' 'Value_1' 'Parameter_2' Value_2' ...")
* ("--save_results_to", type=str, default="results", help="Predicted images and evaluation scores will be saved into this folder")
* ("--n_folds", type=int, default=10, help="Number of folds for n-fold cross validation")
* ("--metrics", type=str, nargs="+", default=["matthews_corrcoef"], help="List of metrics used for validation. Available metrics are %s" % ", ".join(available_scores.keys()))
* ("--features_subset", type=str, default="all", help="If training with just a subset of features, this parameter should be a path to a .txt file containing a list of N features formatted like this: Feature_1\n*Feature_2\n...*Feature_N'. File can contain only 'all', to choose all features")
