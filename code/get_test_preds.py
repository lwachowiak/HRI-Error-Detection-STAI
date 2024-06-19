import numpy as np
import os
import pickle
import json
from data_loader import DataLoader_HRI


def get_full_test_preds(self, model: object, test_X_TS_list: list, interval_length: int, stride_eval: int, model_type: str, dataloader: object, batch_tfms: list = None) -> list:
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
        # TODO get dfs from dataloader

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
    with open(pathprefix + "configs/" + model_to_load + ".json", "r") as f:
        config = json.load(f)

    interval_length = config["interval_length"]
    stride_eval = config["stride_eval"]
    stride_train = config["stride_train"]
    fps = config["fps"]
    label_creation = config["label_creation"]
    task = config["task"]
    rescaling = config["rescaling"]

    # load the test data
    dl = DataLoader_HRI(pathprefix+"data/")
    test_X_TS_list, test_Y_TS_list = dl.get_timeseries_format(interval_length=interval_length, stride_train=stride_train,
                                                              stride_eval=stride_eval, fps=fps, verbose=True, label_creation=label_creation, task=task, rescaling=rescaling)
    test_preds = get_full_test_preds(
        model, test_X_TS_list, interval_length, stride_eval, model_type="Classic", dataloader=dl)
