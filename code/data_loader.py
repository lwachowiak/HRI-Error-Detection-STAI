import os
import pandas as pd
import numpy as np
import re
import math

# TODO:
# - IMPORTANT check how X data is loaded and aligned. it does not seem to work correctly on the last view rows


class DataLoader_HRI:
    """
    Class for loading data from the data folder
    """

    def __init__(self, data_dir: str = "data/", verbose: bool = False):

        self.data_dir = data_dir
        self.verbose = verbose
        self.val_X = []
        self.val_Y = []
        self.train_Y = []
        self.train_X = []
        self.all_X = []
        self.all_Y = []

        openface_data = self.load_data(data_dir+'openface/')
        openpose_data = self.load_data(data_dir+'openpose/')
        opensmile_data = self.load_data(data_dir+'opensmile/')
        speaker_data = self.load_data(data_dir+'speaker_diarization/')
        label_data = self.load_labels(data_dir+'labels/', expand=True)
        self.fold_info = self.load_fold_info(data_dir)
        print("fold_info", self.fold_info)

        # align datastructures
        for filename, df in openpose_data:
            df['frame'] = df['frame_id'].apply(
                lambda x: int(x)+1)  # Convert frame_id to integer and add one
            df.drop(columns=['frame_id'], inplace=True)

        # for filename, df in opensmile_data:
        #    # Convert index to frame number
        #    df['frame'] = (df.index // (100 / 30)).astype(int)+1

        for filename, df in label_data:
            # add column with session number
            df.insert(1, 'session', filename.split('.')[0])

        speaker_data = self.process_speaker_data(speaker_data, label_data)
        # for filename, df in speaker_data:
        #    df["frame"] = (df.index // (100 / 30)).astype(int)+1

        # print the head of the first three dataframes
        if self.verbose:
            inspect_session = 8
            print("\nOpenface data:")
            print(openface_data[inspect_session][1].head(3))
            print(len(openface_data[inspect_session][1]))
            print("\nOpenpose data:")
            print(openpose_data[inspect_session][1].head(3))
            print(len(openpose_data[inspect_session][1]))
            print("\nOpensmile data:")
            print(opensmile_data[inspect_session][1].head(5))
            print(len(opensmile_data[inspect_session][1]))
            print("\nSpeaker data:")
            print(speaker_data[inspect_session][1].head(5))
            print(len(speaker_data[inspect_session][1]))
            print("\nLabel data:")
            print(label_data[inspect_session][1].head(3))
            print(len(label_data[inspect_session][1]))

        # merge data and add to train_X and val_X
        for filename, _ in openface_data:
            merged_df = self.merge_X_data(
                openface_data, openpose_data, opensmile_data, speaker_data, filename)
            if filename.endswith("_train.csv"):
                # self.train_X.append(merged_df) #TODO remove once CV works
                self.all_X.append(merged_df)
            elif filename.endswith("_val.csv"):
                # self.val_X.append(merged_df)
                self.all_X.append(merged_df)
        # labels to train_Y and val_Y
        for filename, df in label_data:
            if filename.endswith("_train.csv"):
                # self.train_Y.append(df)
                self.all_Y.append(df)
            elif filename.endswith("_val.csv"):
                # self.val_Y.append(df)
                self.all_Y.append(df)

        print("\n\nNumber of sessions (data) parsed:",
              len(self.train_X), len(self.val_X), len(self.all_X))
        print("Number of sessions (labels) parsed:",
              len(self.train_Y), len(self.val_Y), len(self.all_Y))

        # concatenate into one single dataframe
        # self.train_X = pd.concat(self.train_X)
        # self.val_X = pd.concat(self.val_X)
        # self.train_Y = pd.concat(self.train_Y)
        # self.val_Y = pd.concat(self.val_Y)
        self.all_X = pd.concat(self.all_X)
        self.all_Y = pd.concat(self.all_Y)

        # print("\n\nNumber of rows in merged dataframes: Train_X:", len(
        #    self.train_X), "Train_Y:", len(self.train_Y), "Val_X:", len(self.val_X), "Val_Y:", len(self.val_Y))

        # COLUMN CLEANUP
        # remove trailing whitespace from column names
        # self.train_X.columns = self.train_X.columns.str.strip()
        # self.val_X.columns = self.val_X.columns.str.strip()
        # self.train_Y.columns = self.train_Y.columns.str.strip()
        # self.val_Y.columns = self.val_Y.columns.str.strip()
        self.all_X.columns = self.all_X.columns.str.strip()
        self.all_Y.columns = self.all_Y.columns.str.strip()

        # for X drop columns with names: 'person_id_openpose', 'week_id_openpose', 'robot_group_openpose', 'end_opensmile', 'start_opensmile' PT: week_id_openpose passes its presence check but fails the drop function, implying it's not in the dataframe
        columns_to_drop = ['person_id_openpose',
                           # 'week_id_openpose',
                           'timestamp_openface',
                           'robot_group_openpose',
                           'end_opensmile',
                           'start_opensmile',
                           'vel_1_x_openpose',
                           'vel_1_y_openpose',
                           'vel_8_x_openpose',
                           'vel_8_y_openpose',
                           'dist_1_8_openpose',
                           'vel_dist_1_8_openpose',
                           'dist_7_0_openpose',
                           'dist_4_0_openpose',
                           'vel_7_x_openpose',
                           'vel_7_y_openpose',
                           'vel_4_x_openpose',
                           'vel_4_y_openpose',
                           'vel_dist_7_0_openpose',
                           'vel_dist_4_0_openpose',
                           'Unnamed: 0_opensmile',
                           'index'
                           ]
        self.exclude_columns(columns_to_drop)

        if self.verbose:
            print("\nMerged data head:")
            # print(self.train_X.head())
            # print(self.val_X.head())
            print(self.all_X.head())
            print("\nMerged data tail:")
            # print(self.train_X.tail())
            # print(self.val_X.tail())
            print(self.all_X.tail())

    @staticmethod
    def extract_file_number(filename: str) -> int:
        '''extract the number from the filename, e.g. 1 from 1_train.csv'''
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else None

    def load_fold_info(self, data_dir: str) -> list:
        """ loads a dict with 1,2,3,4 as keys and the corresponding sessions as values """
        fold_info = {}
        # load from fold_split.csv' with columns "id" and and "fold-subject-independent"
        fold_split = pd.read_csv(data_dir+'fold_split.csv')
        for i in range(1, 5):
            fold_info[i] = fold_split[fold_split['fold-subject-independent']
                                      == i]['id'].values.tolist()
        # ids are strings in the form "10_train" -> just keept the number
        # for i in range(1, 5):
        #    fold_info[i] = [int(re.search(r'\d+', x).group())
        #                    for x in fold_info[i]]
        return fold_info

    def load_data(self, data_dir: str) -> list:
        '''
        load the data from the data_dir into a list of dataframes
        param data_dir: the directory where the data is stored
        return: a list of tuples with the filename and the dataframe
        '''
        data_frames = []
        print(sorted(os.listdir(data_dir), key=self.extract_file_number)
              ) if self.verbose else None

        for filename in sorted(os.listdir(data_dir), key=self.extract_file_number):
            if filename.endswith("_train.csv") or filename.endswith("_val.csv"):
                df = pd.read_csv(os.path.join(data_dir, filename))
                # add session number and df
                data_frames.append((filename, df))
        return data_frames

    def process_speaker_data(self, speaker_data: list, label_data: list, rows_per_second=100) -> list:
        '''
        similar to the labels, the speaker data is originally not presented frame by frame but as time intervals
        '''
        # TODO check output of this function
        i = 0
        data_frames = []
        for filename, df in speaker_data:
            label_df = label_data[i][1]
            i += 1
            frames_count = len(label_df)
            new_data = []
            # initialize the data with speech pauses
            for f in range(1, frames_count+1):
                new_data.append({
                    "robot": 0,
                    "participant": 0,
                    "pause": 1
                })
            # fill in the speaker when appropriate
            for _, row in df.iterrows():
                begin_time = row['start_turn']
                end_time = row['end_turn']
                begin_frame = math.ceil(begin_time * rows_per_second)
                end_frame = math.ceil(end_time * rows_per_second)
                speaker = row['speaker']
                for j in range(begin_frame, end_frame):
                    new_data[j][speaker] = 1
                    if speaker != "pause":
                        new_data[j]["pause"] = 0
            df = pd.DataFrame(new_data)
            data_frames.append((filename, df))
        return data_frames

    def load_labels(self, data_dir: str, expand: bool, rows_per_second: int = 100) -> list:
        '''
        load the labels from the data_dir into a list of dataframes
        '''
        data_frames = []
        print(sorted(os.listdir(data_dir), key=self.extract_file_number))
        for filename in sorted(os.listdir(data_dir), key=self.extract_file_number):
            if filename.endswith("_train.csv") or filename.endswith("_val.csv"):
                df = pd.read_csv(os.path.join(data_dir, filename))
                # change being/end time to frame number, expanding one row to multiple rows based on the duration
                if expand:
                    # create list of right length all 0s
                    # then iterate over the rows and set the right value to 1 in the corresponding rows
                    final_end_time = df['End Time - ss.msec'].iloc[-1]
                    labels_needed = math.ceil(final_end_time * rows_per_second)
                    # 3 labels for labels_needed rows
                    new_data = []
                    for i in range(1, labels_needed+1):
                        new_data.append({
                            "frame": i,
                            "Duration - ss.msec": 1 / rows_per_second,
                            "Begin Time - ss.msec": i / rows_per_second,
                            "UserAwkwardness": 0,
                            "RobotMistake": 0,
                            "InteractionRupture": 0
                        })
                    for _, row in df.iterrows():
                        begin_time = row['Begin Time - ss.msec']
                        end_time = row['End Time - ss.msec']
                        user_awkwardness = row['UserAwkwardness']
                        robot_mistake = row['RobotMistake']
                        interaction_rupture = row['InteractionRupture']
                        begin_frame = math.ceil(begin_time * rows_per_second)
                        end_frame = math.ceil(end_time * rows_per_second)
                        for i in range(begin_frame, end_frame):
                            new_data[i]["UserAwkwardness"] = int(max(
                                new_data[i]["UserAwkwardness"], user_awkwardness))
                            new_data[i]["RobotMistake"] = int(max(
                                new_data[i]["RobotMistake"], robot_mistake))
                            new_data[i]["InteractionRupture"] = int(max(
                                new_data[i]["InteractionRupture"], interaction_rupture))
                            # new_data[i] = {
                            #    "frame": frame_counter,
                            #    "Duration - ss.msec": 1 / rows_per_second,
                            #    "Begin Time - ss.msec": i / rows_per_second,
                            #    "UserAwkwardness": int(user_awkwardness),
                            #    "RobotMistake": int(robot_mistake),
                            #    "InteractionRupture": int(interaction_rupture)
                            # }
                            # frame_counter += 1

                    # for i in range(labels_needed):
                    #     current_time = i / rows_per_second
                    #     try:
                    #         current_df_row = df[(df['Begin Time - ss.msec'] <= current_time) & (
                    #             df['End Time - ss.msec'] >= current_time)]
                    #         user_awkwardness = current_df_row['UserAwkwardness'].values[0]
                    #         robot_mistake = current_df_row['RobotMistake'].values[0]
                    #         interaction_rupture = current_df_row['InteractionRupture'].values[0]
                    #     except:
                    #         print("no row found for session",
                    #               filename, " at time: ", current_time)
                    #         # continue
                    #     new_data.append({
                    #         "frame": frame_counter,
                    #         "Duration - ss.msec": 1 / rows_per_second,
                    #         "Begin Time - ss.msec": current_time,
                    #         "UserAwkwardness": user_awkwardness,
                    #         "RobotMistake": robot_mistake,
                    #         "InteractionRupture": interaction_rupture
                    #     })
                    #     frame_counter += 1
                    df = pd.DataFrame(new_data)
                data_frames.append((filename, df))
        return data_frames

    def merge_X_data(self, openface_data, openpose_data, opensmile_data, speaker_data, filename) -> pd.DataFrame:
        """
        For a specific session (filename), merge the data from the three dataframes.
        """
        df_openface = next(
            df for fname, df in openface_data if fname == filename)
        df_openpose = next(
            df for fname, df in openpose_data if fname == filename)
        df_opensmile = next(
            df for fname, df in opensmile_data if fname == filename)
        df_speaker = next(
            df for fname, df in speaker_data if fname == filename)

        # Ensure all dataframes have 'frame' as the index
        # df_openface.set_index('frame', inplace=True)
        # df_openpose.set_index('frame', inplace=True)
        # df_opensmile.set_index('frame', inplace=True)
        # df_speaker.set_index('frame', inplace=True)

        if self.verbose:
            print(len(df_openface), len(df_openpose),
                  len(df_opensmile), len(df_speaker), filename)

        # Merge the dataframes
        # merge df_speaker and df_opensmile based on index, if one is longer than the other, the extra rows are dropped
        merged_df = df_speaker.add_suffix("_speaker").join(
            df_opensmile.add_suffix('_opensmile'), how='inner')
        merged_df["frame"] = (merged_df.index // (100 / 30)).astype(int)+1
        # merge with the rest based on the frame number
        # merged_df.set_index('frame_opensmile', inplace=True)
        merged_df.set_index('frame', inplace=True)
        df_openpose.set_index('frame', inplace=True)
        df_openface.set_index('frame', inplace=True)

        merged_df = merged_df.join(df_openface.add_suffix('_openface'), how='outer').join(
            df_openpose.add_suffix('_openpose'), how='outer')

        #  drop multiple frame columns
        # columns_to_drop = ['frame_speaker', 'frame_id_openpose']
        # merged_df.drop(columns=columns_to_drop, inplace=True)

        # merged_df = df_openface.add_suffix('_openface').join(
        #    df_openpose.add_suffix('_openpose'), how='outer'
        # ).join(
        #    df_opensmile.add_suffix('_opensmile'), how='outer'
        # ).join(
        #    df_speaker.add_suffix('_speaker'), how='outer'
        # )

        if self.verbose:
            print(len(merged_df), "after merge")

        # Add session id (split at ".")
        merged_df.insert(1, 'session', filename.split('.')[0])

        return merged_df.reset_index()

    def get_summary_format(self, interval_length, stride_train, stride_eval, fps=100, label_creation="full", summary='mean', oversampling_rate=0, undersampling_rate=0, task=2, fold: int = 4) -> tuple:
        """
        Convert the data to summary form. Split the data from the dfs into intervals of length interval_length with stride stride.
        Split takes place of adjacent frames of the same session.
        :param interval_length: The length of the intervals
        :param stride_train: The stride for the training data (oversampling technique)
        :param stride_eval: The stride for the evaluation data (eval update frequency)
        :param fps: The desired fps of the data. Original is 100 fps
        :param label_creation: Either 'full' or 'stride_eval' or 'stride_train'. If 'full' the labels are based on mean of the whole interval, if 'stride' the labels are based on the mean of the stride. This does not affect the final eval but just the optimization goal during training.
        :param summary: The summary type. One of 'mean', 'max', 'min', 'median'
        :param oversampling_rate: x% of the minority class replicated in the training data as oversampling
        :param undersampling_rate: x% of the majority class removed from the training data as undersampling
        :param task: The task to load the data for. 1 for UserAwkwardness, 2 for RobotMistake, 3 for InteractionRupture
        :return: The data in summary format
        """

        val_X_TS, val_Y_summary_list, train_X_TS, train_Y_summary, column_order = self.get_timeseries_format(
            interval_length=interval_length, stride_train=stride_train, stride_eval=stride_eval, fps=fps, label_creation=label_creation, oversampling_rate=oversampling_rate, undersampling_rate=undersampling_rate, task=task, fold=fold)

        if summary not in ['mean', 'max', 'min', 'median']:
            raise ValueError(
                "Summary must be one of 'mean', 'max', 'min', 'median'")

        elif summary == 'mean':
            # squash the timeseries data into one row
            train_X_summary = np.mean(train_X_TS, axis=2)
            val_X_summary_list = [np.mean(val_X_TS[i], axis=2)
                                  for i in range(len(val_X_TS))]

        elif summary == 'max':
            train_X_summary = np.max(train_X_TS, axis=2)
            val_X_summary_list = [np.max(val_X_TS[i], axis=2)
                                  for i in range(len(val_X_TS))]

        elif summary == 'min':
            train_X_summary = np.min(train_X_TS, axis=2)
            val_X_summary_list = [np.min(val_X_TS[i], axis=2)
                                  for i in range(len(val_X_TS))]

        elif summary == 'median':
            train_X_summary = np.median(train_X_TS, axis=2)
            val_X_summary_list = [
                np.median(val_X_TS[i], axis=2) for i in range(len(val_X_TS))]

        # replace NaNs with 0 # TODO do we have this twice??
        train_X_summary = np.nan_to_num(train_X_summary)
        for i in range(len(val_X_summary_list)):
            val_X_summary_list[i] = np.nan_to_num(val_X_summary_list[i])

        return val_X_summary_list, val_Y_summary_list, train_X_summary, train_Y_summary, column_order

    def get_timeseries_format(self, interval_length: int, stride_train: int, stride_eval: int, fps: int = 100, verbose: bool = False, label_creation: str = "full", oversampling_rate: float = 0, undersampling_rate: float = 0, task: int = 2, fold: int = 4) -> tuple:
        """
        Convert the data to timeseries form. Split the data from the dfs into intervals of length interval_length with stride stride.
        Split takes place of adjacent frames of the same session.
        :param interval_length: The length of the intervals
        :param stride_train: The stride for the training data (oversampling technique)
        :param stride_eval: The stride for the evaluation data (eval update frequency)
        :param fps: The desired fps of the data. Original is 100 fps
        :param verbose: Print debug information
        :param label_creation: Either 'full' or 'stride_eval' or 'stride_train'. If 'full' the labels are based on mean of the whole interval, if 'stride' the labels are based on the mean of the stride. This does not affect the final eval but just the optimization goal during training.
        :param oversampling_rate: x% of the minority class replicated in the training data as oversampling
        :param undersampling_rate: x% of the majority class removed from the training data as undersampling
        :param task: The task to load the data for. 1 for UserAwkwardness, 2 for RobotMistake, 3 for InteractionRupture
        :param fold: Fold which the validation data belongs to
        :return: The data in timeseries format and the column order for feature importance analysis
        """
        # get ids based on fold
        val_sessions = self.fold_info[fold]
        train_sessions = []
        for f in self.fold_info:
            if f != fold:
                train_sessions.extend(self.fold_info[f])
        # based on ids, redefine self.train_X and self.val_X
        self.train_X = self.all_X[self.all_X['session'].isin(train_sessions)]
        self.val_X = self.all_X[self.all_X['session'].isin(val_sessions)]
        self.train_Y = self.all_Y[self.all_Y['session'].isin(train_sessions)]
        self.val_Y = self.all_Y[self.all_Y['session'].isin(val_sessions)]

        if verbose:
            print("Train sessions: ", len(train_sessions))
            print("Val sessions: ", len(val_sessions))
            print(self.train_X["session"].unique())
            print(self.val_X["session"].unique())

        num_features = self.train_X.shape[1]
        val_Y_TS_list = []
        val_X_TS_list = []
        train_Y_TS = []
        train_X_TS = []

        cut_length = 10  # drop the last x rows to avoid NaNs TODO think about this

        if label_creation not in ['full', 'stride_eval', 'stride_train']:
            raise ValueError(
                "label_creation must be one of 'full' or 'stride'")

        # Split the data into intervals, if the session changes, start a new interval
        for session in self.train_X['session'].unique():
            session_df = self.train_X[self.train_X['session'] == session]
            # remove session column
            session_df = session_df.drop(columns=['session'])
            column_order = session_df.columns
            # drop last 10 rows to avoid NaNs
            if cut_length > 0:
                session_df = session_df[:-cut_length]
            session_labels = self.train_Y[self.train_Y['session'] == session]
            for i in range(0, len(session_df), stride_train):
                if i + interval_length > len(session_df):
                    # TODO IMPLEMENT PADDING (right now padding is done in eval which might also be ok)
                    break
                interval = session_df.iloc[i:i+interval_length].values.T
                if fps < 100:
                    interval = self.resample(
                        interval=interval, fps=fps, style='mean')
                # for labels use the 3 columns called UserAwkwardness, RobotMistake, InteractionRupture
                labels = session_labels.iloc[i:i+interval_length][[
                    'UserAwkwardness', 'RobotMistake', 'InteractionRupture']].values.T
                # get the 3 majority labels for the interval so it fits the shape
                majority_labels = []
                for label in labels:
                    if label_creation == "full":
                        # get the majority label for the whole interval
                        majority_labels.append(np.argmax(np.bincount(label)))
                    # TODO should i consider stride_eval here?
                    elif label_creation == "stride_train":
                        # get the majority label just for the last stride elements of the interval
                        majority_labels.append(
                            np.argmax(np.bincount(label[-stride_train:])))
                    elif label_creation == "stride_eval":
                        majority_labels.append(
                            np.argmax(np.bincount(label[-stride_eval:])))
                train_X_TS.append(interval)
                train_Y_TS.append(majority_labels)

        # for validation data, stride is equal to interval_length
        for session in self.val_X['session'].unique():
            val_X_TS = []
            val_Y_TS = []
            # if verbose:
            #    print("TS Processing for session: ", session)
            session_df = self.val_X[self.val_X['session'] == session]
            session_df = session_df.drop(columns=['session'])
            # drop last 10 rows to avoid NaNs
            if cut_length > 0:
                session_df = session_df[:-cut_length]
            session_labels = self.val_Y[self.val_Y['session'] == session]
            for i in range(0, len(session_df), stride_eval):  # this was interval_length before
                if i + interval_length > len(session_df):
                    # TODO IMPLEMENT PADDING
                    break
                interval = session_df.iloc[i:i+interval_length].values.T
                if fps < 100:
                    interval = self.resample(
                        interval=interval, fps=fps, style='mean')
                labels = session_labels.iloc[i:i+interval_length][[
                    'UserAwkwardness', 'RobotMistake', 'InteractionRupture']].values.T
                majority_labels = []
                for label in labels:
                    majority_labels.append(np.argmax(np.bincount(label)))
                val_X_TS.append(interval)
                val_Y_TS.append(majority_labels)
            val_X_TS_list.append(val_X_TS)
            val_Y_TS_list.append(val_Y_TS)

        # convert to numpy arrays
        train_X_TS = np.array(train_X_TS)
        train_Y_TS = np.array(train_Y_TS)
        for i in range(len(val_X_TS_list)):
            val_X_TS_list[i] = np.array(val_X_TS_list[i])
            val_Y_TS_list[i] = np.array(val_Y_TS_list[i])

        minority_class = np.argmin(np.bincount(train_Y_TS[:, task]))
        majority_class = np.argmax(np.bincount(train_Y_TS[:, task]))
        if verbose:
            print("Minority class: ", minority_class)
            print("Majority class: ", majority_class)
        if oversampling_rate > 0:  # float indicating the percentage of oversampling # TODO make this work with more than one class
            # oversample the minority class in the training data
            # get the indexes of the minority class
            minority_indexes = np.where(
                train_Y_TS[:, task] == minority_class)[0]
            # oversample the minority class by the oversampling rate
            oversampling_indices = np.random.choice(minority_indexes, int(
                len(minority_indexes) * oversampling_rate), replace=True)
            train_X_TS = np.concatenate(
                (train_X_TS, train_X_TS[oversampling_indices]))
            train_Y_TS = np.concatenate(
                (train_Y_TS, train_Y_TS[oversampling_indices]))
            if verbose:
                print("From minority class: ", len(minority_indexes),
                      " oversampled: ", len(oversampling_indices))
        if undersampling_rate > 0:  # float indicating the percentage of undersampling
            # undersample the majority class in the training data
            # get the indexes of the majority class
            majority_indexes = np.where(
                train_Y_TS[:, task] == majority_class)[0]
            # undersample the majority class by the undersampling rate
            undersampling_indices = np.random.choice(
                majority_indexes, int(len(majority_indexes) * undersampling_rate), replace=False)
            train_X_TS = np.delete(train_X_TS, undersampling_indices, axis=0)
            train_Y_TS = np.delete(train_Y_TS, undersampling_indices, axis=0)
            if verbose:
                print("From majority class: ", len(majority_indexes),
                      " undersampled: ", len(undersampling_indices))

        return val_X_TS_list, val_Y_TS_list, train_X_TS, train_Y_TS, column_order

    def resample(self, interval: list, fps: int, style: str) -> list:
        '''
        Resample the interval to the desired fps. Original framerate is 100 fps
        :param interval: The interval to downsample
        :param fps: The desired fps
        :param style: The style of resampling. One of 'mean', 'max', 'min'
        :return: The downsampled interval
        '''
        # Validate style
        if style not in ['mean', 'max', 'min']:
            raise ValueError("Style must be one of 'mean', 'max', 'min'")
        step = int(100 / fps)
        new_interval = []
        # Iterate over each feature in the interval
        for feature in interval:
            # Convert feature to a NumPy array for vectorized operations
            feature = np.array(feature)
            # Determine the shape of the new downsampled feature
            new_length = len(feature) // step
            reshaped_feature = feature[:new_length * step].reshape(-1, step)
            # Apply the selected downsampling style
            if style == 'mean':
                new_feature = np.mean(reshaped_feature, axis=1)
            elif style == 'max':
                new_feature = np.max(reshaped_feature, axis=1)
            elif style == 'min':
                new_feature = np.min(reshaped_feature, axis=1)
            # Append the downsampled feature to new_interval
            new_interval.append(new_feature.tolist())

        return new_interval

    @ staticmethod
    def impute_nan_with_feature_mean(data) -> np.array:
        for i in range(data.shape[0]):  # Iterate over each sample
            for j in range(data.shape[1]):  # Iterate over each feature
                feature_values = data[i, j, :]
                nan_mask = np.isnan(feature_values)
                if nan_mask.any():
                    feature_mean = np.nanmean(feature_values)
                    feature_values[nan_mask] = feature_mean
        return data

    def exclude_columns(self, columns: list) -> None:
        """
        Exclude columns from the data
        :param columns: The columns to exclude
        """
        for col in columns:
            try:
                self.all_X = self.all_X.drop(columns=col, axis=1)
            except:
                print("Error excluding column with name", col)
            # try:
            #    self.train_X = self.train_X.drop(columns=col, axis=1)
            # except:
            #    print("Error excluding train column with name", col)
            # try:
            #    self.val_X = self.val_X.drop(columns=col, axis=1)
            # except:
            #    print("Error excluding val column with name", col)

    def limit_to_sessions(self, sessions_train: list = None, sessions_val: list = None) -> None:
        """
        Limit the data to the specified sessions
        :param sessions_train: The sessions to include in the training data
        :param sessions_val: The sessions to include in the validation data
        """
        if sessions_train is not None:
            self.train_X = self.train_X[self.train_X['session'].isin(
                sessions_train)]
            self.train_Y = self.train_Y[self.train_Y['session'].isin(
                sessions_train)]
        if sessions_val is not None:
            self.val_X = self.val_X[self.val_X['session'].isin(sessions_val)]
            self.val_Y = self.val_Y[self.val_Y['session'].isin(sessions_val)]


if __name__ == "__main__":
    data_loader = DataLoader_HRI(verbose=True)
    print("\n\n\nData Loaded")

    val_X_ts, val_Y_ts, train_X_ts, train_Y_ts = data_loader.get_timeseries_format(
        interval_length=100, stride_train=100, stride_eval=100, fps=100, label_creation="full")
    print("TS")
    print(len(val_X_ts), len(val_Y_ts), len(train_X_ts), len(train_Y_ts))
    print(train_X_ts.shape, train_Y_ts.shape)

    X_val, Y_val, X_train, Y_train = data_loader.get_summary_format(
        interval_length=100, stride_train=100, stride_eval=100, fps=100, label_creation="full", summary='mean')

    print("X")
    print(len(X_val), len(Y_val), len(X_train), len(Y_train))
    print(X_train.shape, Y_train.shape)

    print(X_val[0].shape)
    print(Y_val[0].shape)
    print(X_val[0])

   # TODO remove

    # def load_labels_old(self, data_dir, expand, rows_per_second=100):
    #     '''
    #     load the labels from the data_dir into a list of dataframes
    #     '''
    #     data_frames = []
    #     for filename in sorted(os.listdir(data_dir), key=self.extract_file_number):
    #         if filename.endswith("_train.csv") or filename.endswith("_val.csv"):
    #             df = pd.read_csv(os.path.join(data_dir, filename))
    #             # print(filename, len(df))
    #             # change being/end time to frame number, expanding one row to multiple rows based on the duration
    #             if expand:
    #                 new_data = []
    #                 frame_counter = 1
    #                 for _, row in df.iterrows():
    #                     begin_time = row['Begin Time - ss.msec']
    #                     end_time = row['End Time - ss.msec']
    #                     total_intervals = math.ceil(
    #                         (end_time - begin_time) * rows_per_second)  # this is the number of frames for the interval
    #                     # print(filename, begin_time, end_time, total_intervals, _) # DEBUG, TODO

    #                     # Generating new time intervals
    #                     # Exclude the last point to avoid overlap
    #                     # new_times = np.linspace(
    #                     #    begin_time, end_time, total_intervals + 1)[:-1]
    #                     # Creating new rows for each time interval
    #                     for t in range(total_intervals):
    #                         new_data.append({
    #                             "frame": frame_counter,
    #                             "Duration - ss.msec": 1 / rows_per_second,
    #                             "Begin Time - ss.msec": begin_time + t / rows_per_second,
    #                             "UserAwkwardness": int(row['UserAwkwardness']),
    #                             "RobotMistake": int(row['RobotMistake']),
    #                             "InteractionRupture": int(row['InteractionRupture'])
    #                         })
    #                         frame_counter += 1

    #                 # Create DataFrame from the list of dictionaries
    #                 df = pd.DataFrame(new_data)
    #                 # print(len(df))
    #             # add session number and df
    #             data_frames.append((filename, df))
    #     return data_frames

    # def resample_old(self, interval, fps, style):
    #     '''
    #     Resample the interval to the desired fps. Original framerate is 100 fps'''
    #     # resample the interval to the desired fps
    #     if style not in ['mean', 'max', 'min']:
    #         raise ValueError("Style must be one of 'mean', 'max', 'min'")
    #     new_interval = []
    #     step = int(100/fps)
    #     for feature in interval:
    #         new_feature = []
    #         # downsample the array by a rate of 100/fps
    #         if style == 'mean':
    #             for i in range(0, len(feature), step):
    #                 new_feature.append(np.mean(feature[i:i+step]))
    #         elif style == 'max':
    #             for i in range(0, len(feature), step):
    #                 new_feature.append(np.max(feature[i:i+step]))
    #         elif style == 'min':
    #             for i in range(0, len(feature), step):
    #                 new_feature.append(np.min(feature[i:i+step]))
    #         new_interval.append(new_feature)
    #     return new_interval
