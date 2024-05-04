# class for loading data from the data/folder

import os
import pandas as pd

# TODO: Add columns for session number


class DataLoader_HRI:
    def __init__(self, data_dir="data/", verbose=False, sampling="Upsample"):

        folders_Y = ["labels/"]

        self.data_dir = data_dir
        self.val_X = []
        self.val_Y = []
        self.train_Y = []
        self.train_X = []

        openface_data = self.load_data(data_dir+'openface/')
        openpose_data = self.load_data(data_dir+'openpose/')
        opensmile_data = self.load_data(data_dir+'opensmile/')

        # align datastructures
        for filename, df in openpose_data:
            df['frame'] = df['frame_id'].apply(
                lambda x: int(x)+1)  # Convert frame_id to integer and add one
            # add column with session number as second column
            print(filename.split('_')[0])
            df.insert(1, 'session', filename.split('_')[0])
            # df['session'] = filename.split('_')[0]

        for filename, df in opensmile_data:
            # Convert index to frame number
            df['frame'] = (df.index // (100 / 30)).astype(int)+1
            # add column with session number
            df.insert(1, 'session', filename.split('_')[0])
            # df['session'] = filename.split('_')[0]

        for filename, df in openface_data:
            # add column with session number
            df.insert(1, 'session', filename.split('_')[0])
            # df['session'] = filename.split('_')[0]

        # print the head of the first three dataframes
        if verbose:
            print("\nOpenface data:")
            print(openface_data[0][1].head(3))
            print("\nOpenpose data:")
            print(openpose_data[-1][1].head(3))
            print("\nOpensmile data:")
            print(opensmile_data[0][1].head(3))

        # merge data
        for filename, _ in openface_data:
            merged_df = self.merge_data(
                openface_data, openpose_data, opensmile_data, filename)
            if filename.endswith("_train.csv"):
                self.train_X.append(merged_df)
            elif filename.endswith("_val.csv"):
                self.val_X.append(merged_df)

        print("\n\nNumber of sessions parsed:",
              len(self.train_X), len(self.val_X))

        # make into one single dataframe
        self.train_X = pd.concat(self.train_X)
        self.val_X = pd.concat(self.val_X)

        if verbose:
            print("\nMerged data head:")
            print(self.train_X.head())
            print(self.val_X.head())
            print("\nMerged data tail:")
            print(self.train_X.tail())
            print(self.val_X.tail())

    def load_data(self, data_dir):
        data_frames = []
        for filename in sorted(os.listdir(data_dir)):
            if filename.endswith("_train.csv") or filename.endswith("_val.csv"):
                df = pd.read_csv(os.path.join(data_dir, filename))
                data_frames.append((filename, df))
        return data_frames

    def merge_data(self, openface_data, openpose_data, opensmile_data, filename):
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

        nan_before_merge = df_openface.isnull().sum().sum() + \
            df_openpose.isnull().sum().sum() + df_opensmile.isnull().sum().sum()
        # Merge the dataframes
        merged_df = df_openface.add_suffix('_openface').join(
            df_openpose.add_suffix('_openpose'), how='outer'
        ).join(
            df_opensmile.add_suffix('_opensmile'), how='outer'
        )
        nan_after_merge = merged_df.isnull().sum().sum()
        # print(
        #    f"NaNs before merge: {nan_before_merge}, NaNs after merge: {nan_after_merge}")

        return merged_df.reset_index()


if __name__ == "__main__":
    data_loader = DataLoader_HRI(verbose=True)
    print("Data Loaded")
    print(data_loader.train_X.head())
