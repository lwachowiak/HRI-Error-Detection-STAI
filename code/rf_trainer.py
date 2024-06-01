from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from data_loader import DataLoader_HRI as DL
import optuna
from optuna.integration import WeightsAndBiasesCallback
import json
from datetime import datetime
import wandb
import numpy as np
from get_metrics import get_metrics
import os

#find path to config files
folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')
class RFTrainer:
    def __init__(self, config):
        self.data_loader = DL("HRI-Error-Detection-STAI/data/")
        self.verbose = True
        self.config = self.read_config(os.path.join(folder, config))
        self.task = self.config["task"]
        self.column_removal_dict = {"REMOVE_NOTHING": ["REMOVE_NOTHING"],
                                    "opensmile": ["opensmile"],
                                    "speaker": ["speaker"],
                                    "openpose": ["openpose"],
                                    "openface": ["openface"],
                                    "openpose, speaker": ["openpose", "speaker"],
                                    "speaker, openpose, openface": ["speaker", "openpose", "openface"]
                                    }
        self.TASK_TO_COLUMN = {0: "UserAwkwardness",
                               1: "RobotMistake", 2: "InteractionRupture"}
    
    def remove_columns(self, columns_to_remove: list, data_X: np.array, column_order: list) -> np.array:
        '''Remove columns from the data.
        :param columns_to_remove: List of columns to remove.
        :param data_X: The data to remove the columns from. Either a list of np.arrays or a np.array.
        :param column_order: The order of the columns in the data.
        :output new_data_X: The data with the specified columns removed.
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
        return new_data_X
    
    def data_from_config(self, config: dict, trial: optuna.Trial):
        """
        create the datasets for training based on the configuration and the trial parameters.
        params: config: dict: The configuration dictionary.
        params: trial: optuna.Trial: The trial object.
        output: tuple: Tuple containing the validation and training datasets.
        """
        data_values = {}
        data_values["interval_length"] = trial.suggest_int("interval_length", **config["interval_length"])
        # strides must be leq than intervallength
        data_values["stride_train"] = trial.suggest_int("stride_train", **config["stride_train"])
        data_values["stride_eval"] = trial.suggest_int("stride_eval", **config["stride_eval"])
        data_values["fps"] = trial.suggest_categorical("fps", config["fps"])
        data_values["columns_to_remove"] = trial.suggest_categorical("columns_to_remove", config["columns_to_remove"])
        columns_to_remove = self.column_removal_dict[data_values["columns_to_remove"]]
        data_values["label_creation"] = trial.suggest_categorical(
            "label_creation", config["label_creation"])
        data_values["nan_handling"] = trial.suggest_categorical(
            "nan_handling", config["nan_handling"])
        data_values["summary"]= trial.suggest_categorical("summary", config["summary"])

        # get timeseries format
        val_X_summary_list, val_Y_summary_list, train_X_summary, train_Y_summary, column_order = self.data_loader.get_summary_format(
            interval_length=data_values["interval_length"], 
            stride_train=data_values["stride_train"], 
            stride_eval=data_values["stride_eval"], 
            fps=data_values["fps"], 
            label_creation=data_values["label_creation"],
            summary=data_values["summary"]
            )
        
        # nan handling
        if data_values["nan_handling"] == "zeros":
            train_X_summary = np.nan_to_num(train_X_summary, nan=0)
            val_X_summary_list = [np.nan_to_num(val_X, nan=0)
                             for val_X in val_X_summary_list]
        if data_values["nan_handling"] == "avg":
            train_X_summary = DL.impute_nan_with_feature_mean(
                train_X_summary)
            val_X_summary_list = [DL.impute_nan_with_feature_mean(
                val_X) for val_X in val_X_summary_list]
            
        # feature removal
        train_X_summary = self.remove_columns(columns_to_remove=columns_to_remove,
                                         data_X=train_X_summary, column_order=column_order)
        val_X_summary_list = self.remove_columns(columns_to_remove=columns_to_remove,
                                            data_X=val_X_summary_list, column_order=column_order)

        train_Y_summary_task = train_Y_summary[:, self.task]

        return val_X_summary_list, val_Y_summary_list, train_X_summary, train_Y_summary, column_order, train_Y_summary_task, data_values

    def get_full_test_preds(self, model: object, val_X_list: list, interval_length: int, stride_eval: int) -> list:
        '''Get full test predictions by repeating the predictions based on intervallength and stride_eval.
        :param model: The model to evaluate.
        :param val_X_TS_list: List of validation/test data per session.
        :param intervallength: The length of the interval to predict.
        :param stride_eval: The stride to evaluate the model.
        '''
        test_preds = []
        for val_X in val_X_list:  # per session
            pred = model.predict(val_X)
            # for each sample in the session, repeat the prediction based on intervallength and stride_eval
            processed_preds = []
            for i, pr in enumerate(pred):
                if i == 0:
                    # first prediction, so append it intervallength times
                    processed_preds.extend([pr]*interval_length)
                else:
                    # all other predictions are appended stride_eval times
                    processed_preds.extend([pr]*stride_eval)
            test_preds.append(processed_preds)
        return test_preds

    def objective(self, trial: optuna.Trial):
        model_params = self.config["model_params"]
        data_params = self.config["data_params"]

        val_X_summary_list, val_Y_summary_list, train_X_summary, train_Y_summary, column_order, train_Y_summary_task, data_values = self.data_from_config(
            data_params, trial)
        
        model_trial_params = {}
        model_trial_params["n_estimators"] = trial.suggest_int("n_estimators", **model_params["n_estimators"])
        model_trial_params["max_depth"] = trial.suggest_int("max_depth", **model_params["max_depth"])
        model_trial_params["random_state"] = trial.suggest_int("random_state", **model_params["random_state"])
        model_trial_params["criterion"] = trial.suggest_categorical(
            "criterion", model_params["criterion"])
        model_trial_params["max_features"] = trial.suggest_categorical(
            "max_features", model_params["max_features"])
        
        model = RandomForestClassifier(**model_trial_params)
        model.fit(train_X_summary, train_Y_summary_task)

        val_preds = self.get_full_test_preds(
            model, val_X_summary_list, data_values["interval_length"], data_values["stride_eval"])
        
        eval_scores = self.get_eval_metrics(val_preds, dataset="val", verbose=False)

        return eval_scores["accuracy"], eval_scores["f1"]

    def read_config(self, file_path):
        """Reads a JSON configuration file and returns the configuration as a dictionary."""
        try:
            with open(file_path, 'r') as file:
                config = json.load(file)
                # print config
                print("\nConfiguration loaded successfully.")
                for key, value in config.items():
                    print(f"{key}: {value}")
                
                return config
        except FileNotFoundError:
            print("\nError: The configuration file was not found.")
        except json.JSONDecodeError:
            print("\nError: The configuration file is not in proper JSON format.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def hyperparam_search(self):
        
        wandb_kwargs = {"project": "HRI-Errors"}
        if True:
            wandbc = WeightsAndBiasesCallback(
                metric_name=["accuracy", "macro f1"], 
                wandb_kwargs=wandb_kwargs)
        else:
            wandbc = None

        date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        study = optuna.create_study(
            directions=["maximize", "maximize"], study_name=date + "_" + "RF")
        print(f"Sampler is {study.sampler.__class__.__name__}")

        study.optimize(self.objective, n_trials=self.config["n_trials"], callbacks=[wandbc])

        trial_with_highest_accuracy = max(
            study.best_trials, key=lambda t: t.values[0])
        
        if self.verbose:
            print("Best trial Accuracy:")
            print(
                "Values: ", trial_with_highest_accuracy.values, "Params: ", trial_with_highest_accuracy.params)
            for key, value in trial_with_highest_accuracy.params.items():
                print(f"    {key}: {value}")

        
        f = "best_{}".format

        for param_name, param_value in trial_with_highest_accuracy.params.items():
            wandb.run.summary[f(param_name)] = param_value

        wandb.run.summary["best accuracy"] = trial_with_highest_accuracy.values[0]
        wandb.run.summary["best macro f1"] = trial_with_highest_accuracy.values[1]

        wandb.finish()

        return study
    
    def get_eval_metrics(self, preds: list, dataset="val", verbose=False) -> dict:
        '''Evaluate model on self.data.val_X and self.data.val_Y. The final missing values in preds are filled with 0s.
        :param preds: List of predictions per session. Per session there is one list of prediction labels.
        :param dataset: The dataset to evaluate on (val or test)
        :output eval_scores: Dictionary containing the evaluation scores (accuracy, macro f1, macro precision, macro recall)
        '''
        y_true = []
        # iterate over all preds (per session) and append 0s if necessary
        for i in range(len(preds)):
            session_id = i
            y_true.append(
                self.data_loader.val_Y[self.data_loader.val_Y['session'] == session_id][self.TASK_TO_COLUMN[self.task]].values)
            # append 0s to preds if necessary
            if len(preds[i]) < len(self.data_loader.val_Y[self.data_loader.val_Y['session'] == session_id]):
                to_append = len(
                    self.data_loader.val_Y[self.data_loader.val_Y['session'] == session_id])-len(preds[i])
                preds[i] = np.append(preds[i], [0]*(to_append))
                if verbose:
                    print("Appended", to_append,
                          "0s to preds for session", session_id, "resulting in", len(preds[i]), "predictions")
            if (len(preds[i]) != len(self.data_loader.val_Y[self.data_loader.val_Y['session'] == session_id])):
                print("ERROR: Length of preds and val_Y do not match for session", session_id,
                      "preds:", len(preds[i]), "val_Y:", len(self.data_loader.val_Y[self.data_loader.val_Y['session'] == session_id]))
        # flatten preds
        preds = np.concatenate(preds)
        y_true = np.concatenate(y_true)
        print("Final preds length", len(preds), len(y_true))

        eval_scores = get_metrics(y_pred=preds, y_true=y_true, tolerance=50)
        print(eval_scores)

        # print confusion matrix
        if verbose:
            print(confusion_matrix(y_true, preds, normalize='all'))
            print(eval_scores)

        return eval_scores

if __name__ == '__main__':
    trainer = RFTrainer('config_rf.json')
    s = trainer.hyperparam_search()