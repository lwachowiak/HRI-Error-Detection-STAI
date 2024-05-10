import os
import pandas as pd
import numpy as np
import re
import math

# TODO:
# - Improve: downsampling
# - Add: data augmentation (TSAI has methods for that)
# - what to do with NANs?
# - IMPORTANT check how X data is loaded an aligned. it does not seem to work correctly on the last view rows


class DataLoader_HRI:
    """
    Class for loading data from the data folder
    """

    def __init__(self, data_dir="data/", verbose=False, sampling="Upsample"):

        self.data_dir = data_dir
        self.val_X = []
        self.val_Y = []
        self.train_Y = []
        self.train_X = []

        openface_data = self.load_data(data_dir+'openface/')
        openpose_data = self.load_data(data_dir+'openpose/')
        opensmile_data = self.load_data(data_dir+'opensmile/')
        label_data = self.load_labels(data_dir+'labels/', expand=True)

        # align datastructures
        for filename, df in openpose_data:
            df['frame'] = df['frame_id'].apply(
                lambda x: int(x)+1)  # Convert frame_id to integer and add one

        for filename, df in opensmile_data:
            # Convert index to frame number
            df['frame'] = (df.index // (100 / 30)).astype(int)+1

        for filename, df in label_data:
            # add column with session number
            df.insert(1, 'session', int(filename.split('_')[0]))

        # print the head of the first three dataframes
        if verbose:
            print(openface_data[0][0])
            print("\nOpenface data:")
            print(openface_data[0][1].head(3))
            print(len(openface_data[0][1]))
            print("\nOpenpose data:")
            print(openpose_data[0][1].head(3))
            print(len(openpose_data[0][1]))
            print("\nOpensmile data:")
            print(opensmile_data[0][1].head(3))
            print(len(opensmile_data[0][1]))
            print("\nLabel data:")
            print(label_data[0][1].head(3))
            print(len(label_data[0][1]))

        # merge data and add to train_X and val_X
        for filename, _ in openface_data:
            merged_df = self.merge_data(
                openface_data, openpose_data, opensmile_data, filename)
            if filename.endswith("_train.csv"):
                self.train_X.append(merged_df)
            elif filename.endswith("_val.csv"):
                self.val_X.append(merged_df)
        # labels to train_Y and val_Y
        for filename, df in label_data:
            if filename.endswith("_train.csv"):
                self.train_Y.append(df)
            elif filename.endswith("_val.csv"):
                self.val_Y.append(df)

        print("\n\nNumber of sessions (data) parsed:",
              len(self.train_X), len(self.val_X))
        print("Number of sessions (labels) parsed:",
              len(self.train_Y), len(self.val_Y))

        # concatenate into one single dataframe
        self.train_X = pd.concat(self.train_X)
        self.val_X = pd.concat(self.val_X)
        self.train_Y = pd.concat(self.train_Y)
        self.val_Y = pd.concat(self.val_Y)

        print("\n\nNumber of rows in merged dataframes: Train_X:", len(
            self.train_X), "Train_Y:", len(self.train_Y), "Val_X:", len(self.val_X), "Val_Y:", len(self.val_Y))

        if verbose:
            print("\nMerged data head:")
            print(self.train_X.head())
            print(self.val_X.head())
            print("\nMerged data tail:")
            print(self.train_X.tail())
            print(self.val_X.tail())

        # COLUMN CLEANUP

        # remove trailing whitespace from column names
        self.train_X.columns = self.train_X.columns.str.strip()
        self.val_X.columns = self.val_X.columns.str.strip()
        self.train_Y.columns = self.train_Y.columns.str.strip()
        self.val_Y.columns = self.val_Y.columns.str.strip()

        # for X drop columns with names: 'person_id_openpose', 'week_id_openpose', 'robot_group_openpose', 'end_opensmile', 'start_opensmile'
        columns_to_drop = ['person_id_openpose', 'week_id_openpose', 'timestamp_openface',
                           'robot_group_openpose', 'end_opensmile', 'start_opensmile',
                           'vel_1_x_openpose', 'vel_1_y_openpose', 'vel_8_x_openpose', 'vel_8_y_openpose', 'dist_1_8_openpose', 'vel_dist_1_8_openpose', 'dist_7_0_openpose', 'dist_4_0_openpose', 'vel_7_x_openpose', 'vel_7_y_openpose', 'vel_4_x_openpose', 'vel_4_y_openpose', 'vel_dist_7_0_openpose', 'vel_dist_4_0_openpose',
                           ]
        self.exclude_columns(columns_to_drop)

    @staticmethod
    def extract_file_number(filename):
        '''extract the number from the filename, e.g. 1 from 1_train.csv'''
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else None

    def load_data(self, data_dir):
        '''
        load the data from the data_dir into a list of dataframes
        '''
        data_frames = []
        print(sorted(os.listdir(data_dir), key=self.extract_file_number))
        for filename in sorted(os.listdir(data_dir), key=self.extract_file_number):
            if filename.endswith("_train.csv") or filename.endswith("_val.csv"):
                df = pd.read_csv(os.path.join(data_dir, filename))
                # add session number and df
                data_frames.append((filename, df))
        return data_frames

    def load_labels(self, data_dir, expand, rows_per_second=100):
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
                    frame_counter = 1
                    final_end_time = df['End Time - ss.msec'].iloc[-1]
                    labels_needed = math.ceil(final_end_time * rows_per_second)
                    # 3 labels for labels_needed rows
                    new_data = []
                    for i in range(labels_needed):
                        new_data.append({
                            "frame": i,
                            "Duration - ss.msec": 1 / rows_per_second,
                            "Begin Time - ss.msec": i / rows_per_second,
                            "UserAwkwardness": 0,
                            "RobotMistake": 0,
                            "InteractionRupture": 0
                        })
                    print("labels needed for session",
                          filename, ":", labels_needed)
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
                            frame_counter += 1

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

    # TODO remove
    def load_labels_old(self, data_dir, expand, rows_per_second=100):
        '''
        load the labels from the data_dir into a list of dataframes
        '''
        data_frames = []
        for filename in sorted(os.listdir(data_dir), key=self.extract_file_number):
            if filename.endswith("_train.csv") or filename.endswith("_val.csv"):
                df = pd.read_csv(os.path.join(data_dir, filename))
                # print(filename, len(df))
                # change being/end time to frame number, expanding one row to multiple rows based on the duration
                if expand:
                    new_data = []
                    frame_counter = 1
                    for _, row in df.iterrows():
                        begin_time = row['Begin Time - ss.msec']
                        end_time = row['End Time - ss.msec']
                        total_intervals = math.ceil(
                            (end_time - begin_time) * rows_per_second)  # this is the number of frames for the interval
                        # print(filename, begin_time, end_time, total_intervals, _) # DEBUG, TODO

                        # Generating new time intervals
                        # Exclude the last point to avoid overlap
                        # new_times = np.linspace(
                        #    begin_time, end_time, total_intervals + 1)[:-1]
                        # Creating new rows for each time interval
                        for t in range(total_intervals):
                            new_data.append({
                                "frame": frame_counter,
                                "Duration - ss.msec": 1 / rows_per_second,
                                "Begin Time - ss.msec": begin_time + t / rows_per_second,
                                "UserAwkwardness": int(row['UserAwkwardness']),
                                "RobotMistake": int(row['RobotMistake']),
                                "InteractionRupture": int(row['InteractionRupture'])
                            })
                            frame_counter += 1

                    # Create DataFrame from the list of dictionaries
                    df = pd.DataFrame(new_data)
                    # print(len(df))
                # add session number and df
                data_frames.append((filename, df))
        return data_frames

    def merge_data(self, openface_data, openpose_data, opensmile_data, filename):
        """
        For a specific session (filename), merge the data from the three dataframes.
        """
        df_openface = next(
            df for fname, df in openface_data if fname == filename)
        df_openpose = next(
            df for fname, df in openpose_data if fname == filename)
        df_opensmile = next(
            df for fname, df in opensmile_data if fname == filename)

        # Ensure all dataframes have 'frame' as the index
        df_openface.set_index('frame', inplace=True)
        df_openpose.set_index('frame', inplace=True)
        df_opensmile.set_index('frame', inplace=True)

        # Merge the dataframes
        merged_df = df_openface.add_suffix('_openface').join(
            df_openpose.add_suffix('_openpose'), how='outer'
        ).join(
            df_opensmile.add_suffix('_opensmile'), how='outer'
        )

        # Add session number
        merged_df.insert(1, 'session', int(filename.split('_')[0]))

        return merged_df.reset_index()

    def get_timeseries_format(self, intervallength, stride_train, stride_eval, fps=100, verbose=False, label_creation="full"):
        """
        Convert the data to timeseries form. Split the data from the dfs into intervals of length intervallength with stride stride.
        Split takes place of adjacent frames of the same session.
        :param intervallength: The length of the intervals
        :param stride_train: The stride for the training data (oversampling technique)
        :param stride_eval: The stride for the evaluation data (eval update frequency)
        :param fps: The desired fps of the data. Original is 100 fps
        :param verbose: Print debug information
        :param eval: Either 'full' or 'stride'. If 'full' the labels are based on mean of the whole interval, if 'stride' the labels are based on the mean of the stride. This does not affect the final eval but just the optimization goal during training.
        :return: The data in timeseries format
        """

        num_features = self.train_X.shape[1]
        val_Y_TS_list = []
        val_X_TS_list = []
        train_Y_TS = []
        train_X_TS = []

        if label_creation not in ['full', 'stride']:
            raise ValueError(
                "label_creation must be one of 'full' or 'stride'")

        # Split the data into intervals, if the session changes, start a new interval
        for session in self.train_X['session'].unique():
            if verbose:
                print("TS Processing for session: ", session)
            session_df = self.train_X[self.train_X['session'] == session]
            session_labels = self.train_Y[self.train_Y['session'] == session]
            for i in range(0, len(session_df), stride_train):
                if i + intervallength > len(session_df):
                    # TODO IMPLEMENT PADDING
                    break
                interval = session_df.iloc[i:i+intervallength].values.T
                if fps < 100:
                    interval = self.resample(
                        interval=interval, fps=fps, style='mean')
                # for labels use the 3 columns called UserAwkwardness, RobotMistake, InteractionRupture
                labels = session_labels.iloc[i:i+intervallength][[
                    'UserAwkwardness', 'RobotMistake', 'InteractionRupture']].values.T
                # get the 3 majority labels for the interval so it fits the shape
                majority_labels = []
                for label in labels:
                    if label_creation == "full":
                        # get the majority label for the whole interval
                        majority_labels.append(np.argmax(np.bincount(label)))
                    elif label_creation == "stride":
                        # get the majority label just for the last stride elements of the interval
                        majority_labels.append(
                            np.argmax(np.bincount(label[-stride_train:])))  # TODO should i consider stride_eval here?
                train_X_TS.append(interval)
                train_Y_TS.append(majority_labels)

        # for validation data, stride is equal to intervallength
        for session in self.val_X['session'].unique():
            val_X_TS = []
            val_Y_TS = []
            if verbose:
                print("TS Processing for session: ", session)
            session_df = self.val_X[self.val_X['session'] == session]
            session_labels = self.val_Y[self.val_Y['session'] == session]
            for i in range(0, len(session_df), stride_eval):  # this was intervallength before
                if i + intervallength > len(session_df):
                    # TODO IMPLEMENT PADDING
                    break
                interval = session_df.iloc[i:i+intervallength].values.T
                if fps < 100:
                    interval = self.resample(
                        interval=interval, fps=fps, style='mean')
                labels = session_labels.iloc[i:i+intervallength][[
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

        return val_X_TS_list, val_Y_TS_list, train_X_TS, train_Y_TS

    def resample(self, interval, fps, style):
        '''
        Resample the interval to the desired fps. Original framerate is 100 fps'''

        # resample the interval to the desired fps
        if style not in ['mean', 'max', 'min']:
            raise ValueError("Style must be one of 'mean', 'max', 'min'")
        new_interval = []
        for feature in interval:
            new_feature = []
            # downsample the array by a rate of 100/fps
            if style == 'mean':
                for i in range(0, len(feature), int(100/fps)):
                    new_feature.append(np.mean(feature[i:i+int(100/fps)]))
            elif style == 'max':
                for i in range(0, len(feature), int(100/fps)):
                    new_feature.append(np.max(feature[i:i+int(100/fps)]))
            elif style == 'min':
                for i in range(0, len(feature), int(100/fps)):
                    new_feature.append(np.min(feature[i:i+int(100/fps)]))
            new_interval.append(new_feature)
        return new_interval

    def exclude_columns(self, columns):
        """
        Exclude columns from the data
        :param columns: The columns to exclude
        """
        self.train_X = self.train_X.drop(columns=columns, axis=1)
        self.val_X = self.val_X.drop(columns=columns, axis=1)

    def limit_to_sessions(self, sessions_train, sessions_val):
        """
        Limit the data to the specified sessions
        :param sessions: The sessions to keep
        """
        self.train_X = self.train_X[self.train_X['session'].isin(
            sessions_train)]
        self.val_X = self.val_X[self.val_X['session'].isin(sessions_val)]
        self.train_Y = self.train_Y[self.train_Y['session'].isin(
            sessions_train)]
        self.val_Y = self.val_Y[self.val_Y['session'].isin(sessions_val)]


if __name__ == "__main__":
    data_loader = DataLoader_HRI(verbose=True)
    print("\n\n\nData Loaded")
    print(data_loader.train_X.head(20))
    print(data_loader.train_Y.head(110))
    print(len(data_loader.train_X), len(data_loader.train_Y))

    # for i in range(100):
    #    print(int(i // (100 / 30)) + 1)
