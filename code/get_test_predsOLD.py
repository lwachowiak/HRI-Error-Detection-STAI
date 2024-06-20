import numpy as np
import os
import pickle
import json
from data_loader import DataLoader_HRI
import pandas as pd
from get_metrics import get_metrics

column_removal_dict = {"REMOVE_NOTHING": ["REMOVE_NOTHING"],
                       "opensmile": ["opensmile"],
                       "speaker": ["speaker"],
                       "openpose": ["openpose"],
                       "openface": ["openface"],
                       "openpose, speaker": ["openpose", "speaker"],
                       "speaker, openpose, openface": ["speaker", "openpose", "openface"],
                       "speaker, openface, opensmile": ["speaker", "openface", "opensmile"],
                       "c_openface": ["c_openface"],
                       "openpose, c_openface": ["openpose", "c_openface"],
                       "vel_dist": ["vel_dist"],
                       "vel_dist, c_openface": ["vel_dist", "c_openface"]
                       }


def remove_columns(columns_to_remove: list, data_X: np.array, column_order: list) -> tuple:
    '''Remove columns from the data.
    :param columns_to_remove: List of columns to remove.
    :param data_X: The data to remove the columns from. Either a list of np.arrays or a np.array.
    :param column_order: The order of the columns in the data.
    :output new_data_X: The data with the specified columns removed and the new column order.
    '''
    # depending on whether data_X is list or np.array
    if isinstance(data_X, list):  # val/test
        new_data_X = [val_X_TS[:, [
            i for i, col in enumerate(column_order)
            if not any(removed_col in col for removed_col in columns_to_remove)
        ]] for val_X_TS in data_X]
    else:  # train
        new_data_X = data_X[:, [
            i for i, col in enumerate(column_order) if not any(removed_col in col for removed_col in columns_to_remove)
        ]]
    new_column_order = [col for col in column_order if not any(
        removed_col in col for removed_col in columns_to_remove)]
    return new_data_X, new_column_order


def get_full_test_preds(model: object, test_X_TS_list: list, interval_length: int, stride_eval: int, model_type: str, batch_tfms: list = None) -> list:
    '''Get full test predictions by repeating the predictions based on interval_length and stride_eval.
    :param model: The model to evaluate.
    :param test_X_TS_list: List of validation/test data per session.
    :param interval_length: The length of the interval to predict.
    :param stride_eval: The stride to evaluate the model.
    :param model_type: Either "Classic" or "TSAI", which have different API calls
    :param batch_tfms: List of batch transformations to apply, if any
    '''

    if model_type not in ["Classic", "TSAI"]:
        raise ValueError(
            "Model type not supported. Parameter model_type must be either 'Classic' or 'TSAI'.")
    test_preds = []
    for session_id, test_X_TS in enumerate(test_X_TS_list):  # per session
        if model_type == "Classic":
            pred = model.predict(test_X_TS)
        elif model_type == "TSAI":
            for tfm in batch_tfms:
                test_X_TS = tfm(test_X_TS)
            valid_probas, valid_targets = model.get_X_preds(
                X=test_X_TS, y=None, with_decoded=False)  # don't use the automatic decoding, there is a bug in the tsai library
            pred = [model.dls.vocab[p]
                    for p in np.argmax(valid_probas, axis=1)]
        # for each sample in the session, repeat the prediction based on interval_length and stride_eval
        processed_preds = []
        for i, pr in enumerate(pred):
            if i == 0:
                # first prediction, so append it interval_length times
                processed_preds.extend([pr]*interval_length)
            else:
                # all other predictions are appended stride_eval times
                processed_preds.extend([pr]*stride_eval)
        test_preds.append(processed_preds)

        # pad with 0s based on max sequence length
        # max_seq_len = len(test_X_TS_list[session_id])
        # if len(processed_preds) < max_seq_len:
        #    test_preds[-1].extend([0]*(max_seq_len-len(processed_preds)))

    return test_preds


if __name__ == "__main__":
    if os.getcwd().endswith("HRI-Error-Detection-STAI"):
        pathprefix = ""
    else:
        pathprefix = "HRI-Error-Detection-STAI/"

    model_to_load = "MiniRocket_2024-06-18-18"

    with open(pathprefix + "code/trained_models/MiniRocketbest_" + model_to_load + ".pkl", "rb") as f:
        model = pickle.load(f)

    # features the model was trained on
    with open(pathprefix + "code/trained_models/MiniRocketbest_" + model_to_load + "_columns.pkl", "rb") as f:
        features = pickle.load(f)

    # load config to get interval_length and stride_eval
    with open(pathprefix + "code/best_model_configs/" + model_to_load + ".json", "r") as f:
        config = json.load(f)

    interval_length = config["data_params"]["interval_length"]
    stride_eval = config["data_params"]["stride_eval"]
    stride_train = config["data_params"]["stride_train"]
    fps = config["data_params"]["fps"]
    label_creation = config["data_params"]["label_creation"]
    task = config["task"]
    rescaling = config["data_params"]["rescaling"]
    columns_to_remove = config["data_params"]["columns_to_remove"]
    columns_to_remove = column_removal_dict[columns_to_remove]

    # load the test data
    dl = DataLoader_HRI(pathprefix+"data/")
    val_X_TS_list, val_Y_TS_list, train_X_TS, train_Y_TS, column_order = dl.get_timeseries_format(interval_length=interval_length, stride_train=stride_train,
                                                                                                  stride_eval=stride_eval, fps=fps, verbose=True, label_creation=label_creation, task=task, rescaling=rescaling, fold=4)
    test_X_TS_list, test_Y_TS_list = dl.get_timeseries_format_test_data(interval_length=interval_length, stride_eval=stride_eval,
                                                                        fps=fps, verbose=True, label_creation=label_creation, task=task, rescaling=rescaling)

    print(len(features))
    print(test_X_TS_list[0].shape)

    # nan handling based on training parameters
    if config["data_params"]["nan_handling"] == "zeros":
        train_X_TS = np.nan_to_num(train_X_TS, nan=0)
        val_X_TS_list = [np.nan_to_num(val_X_TS, nan=0)
                         for val_X_TS in val_X_TS_list]
        test_X_TS_list = [np.nan_to_num(test_X_TS, nan=0)
                          for test_X_TS in test_X_TS_list]
    if config["data_params"]["nan_handling"] == "avg":
        train_X_TS = DataLoader_HRI.impute_nan_with_feature_mean(
            train_X_TS)
        val_X_TS_list = [DataLoader_HRI.impute_nan_with_feature_mean(
            val_X_TS) for val_X_TS in val_X_TS_list]
        test_X_TS_list = [DataLoader_HRI.impute_nan_with_feature_mean(
            test_X_TS) for test_X_TS in test_X_TS_list]

    # feature removal based on training parameters
    train_X_TS, new_column_order = remove_columns(columns_to_remove=columns_to_remove,
                                                  data_X=train_X_TS, column_order=column_order)
    val_X_TS_list, new_column_order = remove_columns(columns_to_remove=columns_to_remove,
                                                     data_X=val_X_TS_list, column_order=column_order)
    print(len(features))
    print(test_X_TS_list[0].shape)
    test_preds = get_full_test_preds(
        model, test_X_TS_list, interval_length=interval_length, stride_eval=stride_eval, model_type="Classic")

    # save the predictions as csv (transposed)
    test_preds_df = pd.DataFrame(test_preds).T
    test_preds_df.to_csv(pathprefix + "code/test_predictions/" +
                         model_to_load + "_test_preds.csv")
