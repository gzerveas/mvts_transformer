from typing import Optional
import os
from multiprocessing import Pool, cpu_count
import glob
import re
import logging
from itertools import repeat, chain

import numpy as np
import pandas as pd
from tqdm import tqdm
from sktime.utils import load_data

from datasets import utils

logger = logging.getLogger('__main__')


class Normalizer(object):
    """
    Normalizes dataframe across ALL contained rows (time steps). Different from per-sample normalization.
    """

    def __init__(self, norm_type, mean=None, std=None, min_val=None, max_val=None):
        """
        Args:
            norm_type: choose from:
                "standardization", "minmax": normalizes dataframe across ALL contained rows (time steps)
                "per_sample_std", "per_sample_minmax": normalizes each sample separately (i.e. across only its own rows)
            mean, std, min_val, max_val: optional (num_feat,) Series of pre-computed values
        """

        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, df):
        """
        Args:
            df: input dataframe
        Returns:
            df: normalized dataframe
        """
        if self.norm_type == "standardization":
            if self.mean is None:
                self.mean = df.mean()
                self.std = df.std()
            return (df - self.mean) / (self.std + np.finfo(float).eps)

        elif self.norm_type == "minmax":
            if self.max_val is None:
                self.max_val = df.max()
                self.min_val = df.min()
            return (df - self.min_val) / (self.max_val - self.min_val + np.finfo(float).eps)

        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            return (df - grouped.transform('mean')) / grouped.transform('std')

        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform('min')
            return (df - min_vals) / (grouped.transform('max') - min_vals + np.finfo(float).eps)

        else:
            raise (NameError(f'Normalize method "{self.norm_type}" not implemented'))


def interpolate_missing(y):
    """
    Replaces NaN values in pd.Series `y` using linear interpolation
    """
    if y.isna().any():
        y = y.interpolate(method='linear', limit_direction='both')
    return y


def subsample(y, limit=256, factor=2):
    """
    If a given Series is longer than `limit`, returns subsampled sequence by the specified integer factor
    """
    if len(y) > limit:
        return y[::factor].reset_index(drop=True)
    return y


class BaseData(object):

    def set_num_processes(self, n_proc):

        if (n_proc is None) or (n_proc <= 0):
            self.n_proc = cpu_count()  # max(1, cpu_count() - 1)
        else:
            self.n_proc = min(n_proc, cpu_count())


class HDD_data(BaseData):
    """
    Dataset class for Hard Drive Disk failure dataset # TODO: INCOMPLETE: does not follow other datasets format
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
    """
    def __init__(self, root_dir, file_list=None, pattern=None, n_proc=1, limit_size=None, config=None):

        self.set_num_processes(n_proc=n_proc)

        self.all_df = self.load_all(root_dir)
        # Sort by serial number and date and index by serial number
        self.all_df = self.all_df.sort_values(by=['serial_number', 'date'])
        self.all_df = self.all_df.set_index('serial_number')

        self.all_IDs = self.all_df.index.unique()  # all asset(disk) IDs (serial numbers)
        self.failed_IDs = self.all_df[
            self.all_df.failure == 1].index.unique()  # IDs corresponding to assets which failed
        self.normal_IDs = sorted(
            list(set(self.all_IDs) - set(self.failed_IDs)))  # IDs corresponding to assets without failure

    def load_all(self, dir_path):
        """
        Loads datasets from all csv files contained in `dir_path` into a dataframe
        Args:
            dir_path: directory containing all individual .csv files. Corresponds to a Quarter

        Returns:
        """
        # each file name corresponds to another date
        input_paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
                       if os.path.isfile(os.path.join(dir_path, f)) and f.endswith('.csv')]

        if self.n_proc > 1:
            # Load in parallel
            _n_proc = min(self.n_proc, len(input_paths))  # no more than file_names needed here
            logger.info("Loading {} datasets files using {} parallel processes ...".format(len(input_paths), _n_proc))
            with Pool(processes=_n_proc) as pool:
                all_df = pd.concat(pool.map(HDD_data.load_single, input_paths))
        else:  # read 1 file at a time
            all_df = pd.concat(HDD_data.load_single(path) for path in input_paths)

        return all_df

    @staticmethod
    def load_single(filepath):
        df = HDD_data.read_data(filepath)
        df = HDD_data.select_columns(df)

        return df

    @staticmethod
    def read_data(filepath):
        """Reads a single .csv, which typically contains a day of datasets of various disks.
        Only Seagate disks are retained."""
        df = pd.read_csv(filepath)
        return df[df['model'].apply(lambda x: x.startswith('ST'))]  # only Seagate models, starting with 'ST', are used

    @staticmethod
    def select_columns(df):
        """Smart9 is the drive's age in hours"""
        df = df.dropna(axis='columns', how='all')  # drop columns containing only NaN
        keep_cols = [col for col in df.columns if 'normalized' not in col]
        df = df[keep_cols]
        return df

    @staticmethod
    def process_columns(df):

        df['date'] = pd.to_datetime(df['date'])
        df['failure'] = df['failure'].astype(bool)
        df[['capacity_bytes', 'model']] = df[['capacity_bytes', 'model']].astype('category')

        return df


class WeldData(BaseData):
    """
    Dataset class for welding dataset.
    Attributes:
        all_df: dataframe indexed by ID, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, root_dir, file_list=None, pattern=None, n_proc=1, limit_size=None, config=None):

        self.set_num_processes(n_proc=n_proc)

        self.all_df = self.load_all(root_dir, file_list=file_list, pattern=pattern)
        self.all_df = self.all_df.sort_values(by=['weld_record_index'])  # datasets is presorted
        # TODO: There is a single ID that causes the model output to become nan - not clear why
        self.all_df = self.all_df[self.all_df['weld_record_index'] != 920397]  # exclude particular ID
        self.all_df = self.all_df.set_index('weld_record_index')
        self.all_IDs = self.all_df.index.unique()  # all sample (session) IDs
        self.max_seq_len = 66
        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        self.feature_names = ['wire_feed_speed', 'current', 'voltage', 'motor_current', 'power']
        self.feature_df = self.all_df[self.feature_names]

    def load_all(self, root_dir, file_list=None, pattern=None):
        """
        Loads datasets from csv files contained in `root_dir` into a dataframe, optionally choosing from `pattern`
        Args:
            root_dir: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_dir` to consider.
                Otherwise, entire `root_dir` contents will be used.
            pattern: optionally, apply regex string to select subset of files
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
        """
        # each file name corresponds to another date. Also tools (A, B) and others.

        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_dir, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_dir, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_dir, '*')))

        if pattern is None:
            # by default evaluate on
            selected_paths = data_paths
        else:
            selected_paths = list(filter(lambda x: re.search(pattern, x), data_paths))

        input_paths = [p for p in selected_paths if os.path.isfile(p) and p.endswith('.csv')]
        if len(input_paths) == 0:
            raise Exception("No .csv files found using pattern: '{}'".format(pattern))

        if self.n_proc > 1:
            # Load in parallel
            _n_proc = min(self.n_proc, len(input_paths))  # no more than file_names needed here
            logger.info("Loading {} datasets files using {} parallel processes ...".format(len(input_paths), _n_proc))
            with Pool(processes=_n_proc) as pool:
                all_df = pd.concat(pool.map(WeldData.load_single, input_paths))
        else:  # read 1 file at a time
            all_df = pd.concat(WeldData.load_single(path) for path in input_paths)

        return all_df

    @staticmethod
    def load_single(filepath):
        df = WeldData.read_data(filepath)
        df = WeldData.select_columns(df)
        num_nan = df.isna().sum().sum()
        if num_nan > 0:
            logger.warning("{} nan values in {} will be replaced by 0".format(num_nan, filepath))
            df = df.fillna(0)

        return df

    @staticmethod
    def read_data(filepath):
        """Reads a single .csv, which typically contains a day of datasets of various weld sessions.
        """
        df = pd.read_csv(filepath)
        return df

    @staticmethod
    def select_columns(df):
        """"""
        df = df.rename(columns={"per_energy": "power"})
        # Sometimes 'diff_time' is not measured correctly (is 0), and power ('per_energy') becomes infinite
        is_error = df['power'] > 1e16
        df.loc[is_error, 'power'] = df.loc[is_error, 'true_energy'] / df['diff_time'].median()

        df['weld_record_index'] = df['weld_record_index'].astype(int)
        keep_cols = ['weld_record_index', 'wire_feed_speed', 'current', 'voltage', 'motor_current', 'power']
        df = df[keep_cols]

        return df


class TSRegressionArchive(BaseData):
    """
    Dataset class for datasets included in:
        1) the Time Series Regression Archive (www.timeseriesregression.org), or
        2) the Time Series Classification Archive (www.timeseriesclassification.com)
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, root_dir, file_list=None, pattern=None, n_proc=1, limit_size=None, config=None):

        #self.set_num_processes(n_proc=n_proc)

        self.config = config

        self.all_df, self.labels_df = self.load_all(root_dir, file_list=file_list, pattern=pattern)
        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

    def load_all(self, root_dir, file_list=None, pattern=None):
        """
        Loads datasets from csv files contained in `root_dir` into a dataframe, optionally choosing from `pattern`
        Args:
            root_dir: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_dir` to consider.
                Otherwise, entire `root_dir` contents will be used.
            pattern: optionally, apply regex string to select subset of files
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """

        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_dir, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_dir, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_dir, '*')))

        if pattern is None:
            # by default evaluate on
            selected_paths = data_paths
        else:
            selected_paths = list(filter(lambda x: re.search(pattern, x), data_paths))

        input_paths = [p for p in selected_paths if os.path.isfile(p) and p.endswith('.ts')]
        if len(input_paths) == 0:
            raise Exception("No .ts files found using pattern: '{}'".format(pattern))

        all_df, labels_df = self.load_single(input_paths[0])  # a single file contains dataset

        return all_df, labels_df

    def load_single(self, filepath):

        # Every row of the returned df corresponds to a sample;
        # every column is a pd.Series indexed by timestamp and corresponds to a different dimension (feature)
        if self.config['task'] == 'regression':
            df, labels = utils.load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True, replace_missing_vals_with='NaN')
            labels_df = pd.DataFrame(labels, dtype=np.float32)
        elif self.config['task'] == 'classification':
            df, labels = load_data.load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True, replace_missing_vals_with='NaN')
            labels = pd.Series(labels, dtype="category")
            self.class_names = labels.cat.categories
            labels_df = pd.DataFrame(labels.cat.codes, dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss
        else:  # e.g. imputation
            try:
                df, labels = load_data.load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                                     replace_missing_vals_with='NaN')
            except:
                df, _ = utils.load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                                 replace_missing_vals_with='NaN')
            labels_df = None

        lengths = df.applymap(lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series
        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        # most general check: len(np.unique(lengths.values)) > 1:  # returns array of unique lengths of sequences
        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            logger.warning("Not all time series dimensions have same length - will attempt to fix by subsampling first dimension...")
            df = df.applymap(subsample)  # TODO: this addresses a very specific case (PPGDalia)

        if self.config['subsample_factor']:
            df = df.applymap(lambda x: subsample(x, limit=0, factor=self.config['subsample_factor']))

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
            logger.warning("Not all samples have same length: maximum length set to {}".format(self.max_seq_len))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)
        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0]*[row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df


class SemicondTraceData(BaseData):
    """
    Dataset class for semiconductor manufacturing sensor trace data.
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """
    # TODO: currently all *numeric* features which are not *dataset-wise* constant are kept. Sample-wise constants are included
    features = ['Actual Bias voltage (AT/CH2/RFGen/RFMatch.rMatchBias)', 'Actual Pressure (AT/CH2/PressCtrl.rPress)',
                'Ampoule wafer count (AT/CH2/Gaspanel/Stick01/BUBBLER.cAmpouleWaferCount)',
                'Ampoule wafer count (AT/CH2/Gaspanel/Stick05/BUBBLER.cAmpouleWaferCount)',
                'Backside Flow Reading (AT/CH2/VacChuck.rBacksideFlow)',
                'Backside Pressure Reading (AT/CH2/VacChuck.rBacksidePress)',
                'Backside Pressure Setpoint (AT/CH2/VacChuck.wBacksidePressSP)',
                'Bubbler ampoule accumulated flow (AT/CH2/Gaspanel/Stick01/BUBBLER.cAmpouleLifeAccFlow)',
                'Bubbler ampoule accumulated flow (AT/CH2/Gaspanel/Stick05/BUBBLER.cAmpouleLifeAccFlow)',
                'Current Flow (AT/CH2/Gaspanel/Stick01.rFlow)', 'Current Flow (AT/CH2/Gaspanel/Stick01/Mfc.rFlow)',
                'Current Flow (AT/CH2/Gaspanel/Stick02.rFlow)', 'Current Flow (AT/CH2/Gaspanel/Stick02/Mfc.rFlow)',
                'Current Flow (AT/CH2/Gaspanel/Stick03.rFlow)', 'Current Flow (AT/CH2/Gaspanel/Stick03/Mfc.rFlow)',
                'Current Flow (AT/CH2/Gaspanel/Stick05.rFlow)', 'Current Flow (AT/CH2/Gaspanel/Stick05/Mfc.rFlow)',
                'Current Flow (AT/CH2/Gaspanel/Stick06.rFlow)', 'Current Flow (AT/CH2/Gaspanel/Stick06/Mfc.rFlow)',
                'Current Flow (AT/CH2/Gaspanel/Stick09.rFlow)', 'Current Flow (AT/CH2/Gaspanel/Stick09/Mfc.rFlow)',
                'Current Flow (AT/CH2/Gaspanel/Stick21.rFlow)', 'Current Flow (AT/CH2/Gaspanel/Stick21/Mfc.rFlow)',
                'Current Flow (AT/CH2/Gaspanel/Stick22.rFlow)', 'Current Flow (AT/CH2/Gaspanel/Stick22/Mfc.rFlow)',
                'Current Position SP Percent (AT/CH2/PressCtrl.rPosSPP)', 'Current Power SP (AT/CH2/RFGen.rPowerSP)',
                'Current Pressure in PSI (AT/CH2/Gaspanel/Stick01/Transducer.rPressure)',
                'Current Pressure in PSI (AT/CH2/Gaspanel/Stick05/Transducer.rPressure)',
                'Current Pressure in PSI (AT/CH2/Gaspanel/Stick08/Transducer.rPressure)',
                'Current Pressure in PSI (AT/CH2/Gaspanel/Stick09/Transducer.rPressure)',
                'Current Pressure in Torr (AT/CH2/Gaspanel/Stick01/Transducer.rPressureTorr)',
                'Current Pressure in Torr (AT/CH2/Gaspanel/Stick05/Transducer.rPressureTorr)',
                'Current Pressure in Torr (AT/CH2/Gaspanel/Stick08/Transducer.rPressureTorr)',
                'Current Pressure in Torr (AT/CH2/Gaspanel/Stick09/Transducer.rPressureTorr)',
                'Current Recipe Count (AT/CH2/Clean/Idle Purge.CurRcpCnt)',
                'Current Recipe Count (AT/CH2/Clean/On Load Clean.CurRcpCnt)',
                'Current recipe step number (AT/CH2.@RecipeStep01)',
                'Current servo error  (AT/CH2/TempCtrl/Heater.rOutputCurrServoError)',
                'Cycle Count (AT/CH2/Gaspanel/Stick01/Service/Cycle Purge By Pressure.cnfCycleCount)',
                'Cycle Count (AT/CH2/Gaspanel/Stick01/Service/Cycle Purge By Time.cnfCycleCount)',
                'Cycle Count (AT/CH2/Gaspanel/Stick05/Service/Cycle Purge By Pressure.cnfCycleCount)',
                'Cycle Count (AT/CH2/Gaspanel/Stick05/Service/Cycle Purge By Time.cnfCycleCount)',
                'Cycle Count (AT/CH2/Gaspanel/Stick08/Service/Cycle Purge By Pressure.cnfCycleCount)',
                'Default temperature setpoint (AT/CH2/Watlow1/Ch_1.cDefaultSetpoint)',
                'Default temperature setpoint (AT/CH2/Watlow1/Ch_2.cDefaultSetpoint)',
                'Default temperature setpoint (AT/CH2/Watlow1/Ch_6.cDefaultSetpoint)',
                'Default temperature setpoint (AT/CH2/Watlow2/Ch_4.cDefaultSetpoint)',
                'Default temperature setpoint (AT/CH2/Watlow2/Ch_5.cDefaultSetpoint)',
                'Estimated Ampoule wafer count (AT/CH2/Gaspanel/Stick05/BUBBLER.cEstAmpouleWaferCount)',
                'Expected Lid Heater Temperature (AT/CH2/Rcp.wHdrLidHtrTemp)',
                'Final Leak Check pressure (AT/CH2/Services/CVDLeakCheck/LeakCheck.rLeakCheckFinalPressure)',
                'Final Leak Rate (AT/CH2/Services/CVDLeakCheck/LeakCheck.rFinalLeakRate)',
                'Flow Setpoint (AT/CH2/Gaspanel/Stick02/Mfc.wSetpoint)',
                'Flow Setpoint (AT/CH2/Gaspanel/Stick03/Mfc.wSetpoint)',
                'Flow Setpoint (AT/CH2/Gaspanel/Stick05/Mfc.wSetpoint)',
                'Flow Setpoint (AT/CH2/Gaspanel/Stick06/Mfc.wSetpoint)',
                'Flow Setpoint (AT/CH2/Gaspanel/Stick09/Mfc.wSetpoint)',
                'Flow Setpoint (AT/CH2/Gaspanel/Stick21/Mfc.wSetpoint)',
                'Flow Setpoint (AT/CH2/Gaspanel/Stick22/Mfc.wSetpoint)',
                'Next wafer slot, side 1 (AT/CH2.@NextCassSlot01_01)',
                'Next wafer src, side 1 (AT/CH2.@NextCassId01_01)', 'Temp Reading  (AT/CH2/Watlow1/Ch_1.rTempReading)',
                'Temp Reading  (AT/CH2/Watlow1/Ch_2.rTempReading)', 'Temp Reading  (AT/CH2/Watlow1/Ch_3.rTempReading)',
                'Temp Reading  (AT/CH2/Watlow1/Ch_4.rTempReading)', 'Temp Reading  (AT/CH2/Watlow1/Ch_5.rTempReading)',
                'Temp Reading  (AT/CH2/Watlow1/Ch_6.rTempReading)', 'Temp Reading  (AT/CH2/Watlow1/Ch_7.rTempReading)',
                'Temp Reading  (AT/CH2/Watlow1/Ch_8.rTempReading)', 'Temp Reading  (AT/CH2/Watlow2/Ch_1.rTempReading)',
                'Temp Reading  (AT/CH2/Watlow2/Ch_2.rTempReading)', 'Temp Reading  (AT/CH2/Watlow2/Ch_3.rTempReading)',
                'Temp Reading  (AT/CH2/Watlow2/Ch_4.rTempReading)', 'Temp Reading  (AT/CH2/Watlow2/Ch_5.rTempReading)',
                'Temp Reading  (AT/CH2/Watlow2/Ch_6.rTempReading)', 'Temp Reading  (AT/CH2/Watlow2/Ch_8.rTempReading)',
                'Temp Reading  (AT/CH2/Watlow3/Ch_1.rTempReading)', 'Temp Reading  (AT/CH2/Watlow3/Ch_2.rTempReading)',
                'Temp Reading  (AT/CH2/Watlow3/Ch_3.rTempReading)', 'Temp Reading  (AT/CH2/Watlow3/Ch_4.rTempReading)',
                'Temp Reading  (AT/CH2/Watlow3/Ch_5.rTempReading)', 'Temp Reading  (AT/CH2/Watlow3/Ch_6.rTempReading)',
                'Temp Reading  (AT/CH2/Watlow3/Ch_7.rTempReading)']

    def __init__(self, root_dir, file_list=None, pattern=None, n_proc=8, limit_size=None, config=None):

        self.set_num_processes(n_proc=n_proc)

        ## Get labels
        wafer_measurements_path = os.path.join(root_dir, "waferdata/")
        logger.info("Getting wafer measurements ...")
        # Dataframe which holds all measurements: mean thickness (and deposition rate for the subset "type 1"), roughness (std of thickness)
        measurements_df = self.get_measurements(wafer_measurements_path)

        ## Get metadata (e.g. mapping between measurement file and trace file)
        catalog_path = os.path.join(root_dir, "CTF03.catalog.20200629.csv")
        logger.info("Getting wafer metadata ...")
        # This dataframe holds for all wafers all metadata per wafer (including measurements, when they exist, and corresponding trace file)
        metadata_df = self.get_metadata(catalog_path, measurements_df)

        # TODO: select subset here (e.g. here 20A wafers selected), or set file_list=None to use all
        files_20A = metadata_df.loc[metadata_df['ChamberRecipeID'] == 'QUALCH2CO20', 'TraceDataFile']
        IDs_20A = files_20A.index
        files_20A = list(map(self.convert_tracefilename, files_20A))

        ## Get trace files
        tracedata_dir = os.path.join(root_dir, "tracedata/CTF03_CH2_QUALCH2CO_CH2_G0009")
        logger.info("Getting sensor trace data ...")
        self.all_df = self.load_all(tracedata_dir, file_list=files_20A, pattern=pattern)

        self.all_IDs = self.all_df.index.unique()  # all sample (session) IDs

        # TODO: select prediction objective here: any of ['Mean_dep_rate', 'std_thickness', 'mean_thickness']
        if config['task'] == 'regression':
            labels_col = config['labels'] if config['labels'] else 'Mean_dep_rate'
            self.labels_df = pd.DataFrame(metadata_df.loc[self.all_IDs, labels_col], dtype=np.float32)
            self.labels_df = self.labels_df[~self.labels_df[labels_col].isna()]
            self.all_IDs = self.labels_df.index
            self.all_df = self.all_df.loc[self.all_IDs]

        self.max_seq_len = 130  # TODO: for 20A

        if (limit_size is not None) and (limit_size < len(self.all_IDs)):
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        self.feature_names = SemicondTraceData.features
        self.feature_df = self.all_df[self.feature_names]

        # Replace NaN values (at this point due to some columns missing from some trace files)
        if self.feature_df.isna().any().any():
            self.feature_df = self.feature_df.fillna(0)

        return

    def make_pjid(self, toolID, pjID):
        """Convert PJID format of catalog file to the one used in measurement files"""
        return toolID + '-' + pjID.split('.')[0]

    def convert_tracefilename(self, filepath):
        """
        This processing depends on how tracefiles are stored (flat directory hierarchy or not, .csv or .zip)
        See retrieve_tracefiles.py for options.
        Here, a flat hierarchy and .csv format is assumed
        """
        filename, extension = os.path.splitext(os.path.basename(filepath))
        return filename + '.csv'

    def get_measurements(self, wafer_measurements_path):

        # There are 2 "types" of files, the ones that start with "Rate" and contain mean deposition rate (type 1)
        # and the ones that start with "mCTF" (type 2),
        # which only contain mean thickness and fewer columns with different ID and measurement column name.
        deprate_df1 = self.load_all(wafer_measurements_path, pattern="Rate_time_series.*_Average_", mode='simple')
        deprate_df1 = deprate_df1.rename(columns={"Mea_value": "mean_thickness"})

        deprate_df2 = self.load_all(wafer_measurements_path, pattern=r"/mCTF.*_Average_", mode='simple')
        deprate_df2 = deprate_df2.rename(columns={"Wafer_mean": "mean_thickness"})

        # Merge the 2 types for deposition rate/thickness
        deprate_df = pd.merge(deprate_df1, deprate_df2, how='outer', left_on=['Proc_cj_id', 'Wafer_id'],
                              right_on=['Control_job_id', 'Wafer_id'],
                              left_index=False, right_index=False, sort=True,
                              suffixes=(None, '_right'), copy=True, indicator=True,
                              validate=None)
        # The 2 types contain overlapping sets of wafers, so we need to form a common measurement column
        right_only = deprate_df['mean_thickness'].isnull()
        deprate_df.loc[right_only, 'mean_thickness'] = deprate_df.loc[right_only, 'mean_thickness_right']

        # Repeat process for roughness (std of thickness) measurement files
        roughness_df1 = self.load_all(wafer_measurements_path, pattern="Rate_time_series.*_StdDev_", mode='simple')
        roughness_df1 = roughness_df1.rename(columns={"Std_dep_thk": "std_thickness"})

        roughness_df2 = self.load_all(wafer_measurements_path, pattern=r"/mCTF.*_StdDev_", mode='simple')
        roughness_df2 = roughness_df2.rename(columns={"Wafer_std": "std_thickness"})

        roughness_df = pd.merge(roughness_df1, roughness_df2, how='outer', left_on=['Proc_cj_id', 'Wafer_id'],
                                right_on=['Control_job_id', 'Wafer_id'],
                                left_index=False, right_index=False, sort=True,
                                suffixes=(None, '_right'), copy=True, indicator=True,
                                validate=None)

        right_only = roughness_df['std_thickness'].isnull()
        roughness_df.loc[right_only, 'std_thickness'] = roughness_df.loc[right_only, 'std_thickness_right']

        # Dataframe which holds all measurements: mean thickness (and deposition rate for the subset "type 1"), roughness (std of thickness)
        measurements_df = pd.merge(deprate_df, roughness_df, how='inner', on=['Proc_cj_id', 'Wafer_id'],
                                   left_index=False, right_index=False, sort=True,
                                   suffixes=('_x', '_y'), copy=True, indicator=False,
                                   validate=None)

        assert sum(measurements_df.mean_thickness.isnull()) == 0, "Missing thickness measurements"
        assert sum(measurements_df.std_thickness.isnull()) == 0, "Missing roughness measurements"

        return measurements_df

    def get_metadata(self, catalog_path, measurements_df):

        catalog_df = pd.read_csv(catalog_path)
        # Restrict to Chamber 2
        catalog_df = catalog_df[catalog_df['ChamberID'] == 'CH2']
        # Restrict to the recipes corresponding to existing measurememt wafers and associated product wafers
        catalog_df = catalog_df[catalog_df['ChamberRecipeID'].isin(['QUALCH2CO20', 'QUALCH2CO100', 'CH2_G0009'])]
        catalog_df['pjid'] = catalog_df[['ToolID', 'PJID']].apply(lambda x: self.make_pjid(*x), axis=1)

        # This dataframe holds all metadata per wafer (including measurements, when they exist, and corresponding trace file)
        metadata_df = pd.merge(catalog_df, measurements_df, how='left', left_on=['pjid', 'WaferID'],
                               right_on=['Proc_cj_id', 'Wafer_id'],
                               left_index=False, right_index=False, sort=True,
                               suffixes=('_x', '_y'), copy=True, indicator=False,
                               validate=None)
        metadata_df = metadata_df.set_index('WaferPassID')

        return metadata_df

    def load_all(self, root_dir, file_list=None, pattern=None, mode=None):
        """
        Loads datasets from csv files contained in `root_dir` into a dataframe, optionally choosing from `pattern`
        Args:
            root_dir: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_dir` to consider.
                Otherwise, entire `root_dir` contents will be used.
            pattern: optionally, apply regex string to select subset of files
            func: function to use for loading a single file
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
        """

        # if func is None:
        #     func = SemicondTraceData.load_single

        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_dir, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_dir, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_dir, '*')))

        if pattern is None:
            # by default evaluate on
            selected_paths = data_paths
        else:
            selected_paths = list(filter(lambda x: re.search(pattern, x), data_paths))

        input_paths = [p for p in selected_paths if os.path.isfile(p) and p.endswith('.csv')]
        if len(input_paths) == 0:
            raise Exception("No .csv files found using pattern: '{}'".format(pattern))

        if (mode != 'simple') and (self.n_proc > 1):
            # Load in parallel
            _n_proc = min(self.n_proc, len(input_paths))  # no more than file_names needed here
            logger.info("Loading {} datasets files using {} parallel processes ...".format(len(input_paths), _n_proc))
            with Pool(processes=_n_proc) as pool:
                # done like this because multiprocessing needs the *explicit* function call
                # and not a reference to a function, e.g. func = pd.read_csv
                all_df = pd.concat(pool.map(SemicondTraceData.load_single, input_paths))
        else:  # read 1 file at a time
            if mode == 'simple':
                all_df = pd.concat(pd.read_csv(path) for path in tqdm(input_paths))
            else:
                all_df = pd.concat(SemicondTraceData.load_single(path) for path in tqdm(input_paths))

        return all_df

    @staticmethod
    def load_single(filepath):
        df = SemicondTraceData.read_data(filepath)
        df = SemicondTraceData.select_columns(df)

        df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
        df = df.sort_values(by=['WaferPassID', 'TimeStamp'])
        df = df.set_index('WaferPassID')

        # Replace NaN values (at this point, these are missing values in a variable/sequence)
        feat_col = [col for col in df.columns if col in SemicondTraceData.features]  # because some columns are missing in some tracefiles
        if df[feat_col].isna().any().any():
            grp = df.groupby(by=df.index)
            df.loc[:, feat_col] = grp.transform(interpolate_missing)

        return df

    @staticmethod
    def read_data(filepath):
        """Reads a single .csv, which typically contains a day of datasets of various weld sessions.
        """
        df = pd.read_csv(filepath)
        return df

    @staticmethod
    def select_columns(df):

        # Kept just as an example
        # df = df.rename(columns={"per_energy": "power"})
        # # Sometimes 'diff_time' is not measured correctly (is 0), and power ('per_energy') becomes infinite
        # is_error = df['power'] > 1e16
        # df.loc[is_error, 'power'] = df.loc[is_error, 'true_energy'] / df['diff_time'].median()
        # keep_cols = ['weld_record_index', 'wire_feed_speed', 'current', 'voltage', 'motor_current', 'power']
        # df = df[keep_cols]

        # This doesn't work because some columns are missing in some tracefiles
        # keep_cols = ['WaferPassID', 'TimeStamp'] + SemicondTraceData.features
        # df = df[keep_cols]

        return df


class PMUData(BaseData):
    """
    Dataset class for Phasor Measurement Unit dataset.
    Attributes:
        all_df: dataframe indexed by ID, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        max_seq_len: maximum sequence (time series) length (optional). Used only if script argument `max_seq_len` is not
            defined.
    """

    def __init__(self, root_dir, file_list=None, pattern=None, n_proc=1, limit_size=None, config=None):

        self.set_num_processes(n_proc=n_proc)

        self.all_df = self.load_all(root_dir, file_list=file_list, pattern=pattern)

        if config['data_window_len'] is not None:
            self.max_seq_len = config['data_window_len']
            # construct sample IDs: 0, 0, ..., 0, 1, 1, ..., 1, 2, ..., (num_whole_samples - 1)
            # num_whole_samples = len(self.all_df) // self.max_seq_len  # commented code is for more general IDs
            # IDs = list(chain.from_iterable(map(lambda x: repeat(x, self.max_seq_len), range(num_whole_samples + 1))))
            # IDs = IDs[:len(self.all_df)]  # either last sample is completely superfluous, or it has to be shortened
            IDs = [i // self.max_seq_len for i in range(self.all_df.shape[0])]
            self.all_df.insert(loc=0, column='ExID', value=IDs)
        else:
            # self.all_df = self.all_df.sort_values(by=['ExID'])  # dataset is presorted
            self.max_seq_len = 30

        self.all_df = self.all_df.set_index('ExID')
        # rename columns
        self.all_df.columns = [re.sub(r'\d+', str(i//3), col_name) for i, col_name in enumerate(self.all_df.columns[:])]
        #self.all_df.columns = ["_".join(col_name.split(" ")[:-1]) for col_name in self.all_df.columns[:]]
        self.all_IDs = self.all_df.index.unique()  # all sample (session) IDs

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        self.feature_names = self.all_df.columns  # all columns are used as features
        self.feature_df = self.all_df[self.feature_names]

    def load_all(self, root_dir, file_list=None, pattern=None):
        """
        Loads datasets from csv files contained in `root_dir` into a dataframe, optionally choosing from `pattern`
        Args:
            root_dir: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_dir` to consider.
                Otherwise, entire `root_dir` contents will be used.
            pattern: optionally, apply regex string to select subset of files
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
        """

        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_dir, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_dir, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_dir, '*')))

        if pattern is None:
            # by default evaluate on
            selected_paths = data_paths
        else:
            selected_paths = list(filter(lambda x: re.search(pattern, x), data_paths))

        input_paths = [p for p in selected_paths if os.path.isfile(p) and p.endswith('.csv')]
        if len(input_paths) == 0:
            raise Exception("No .csv files found using pattern: '{}'".format(pattern))

        if self.n_proc > 1:
            # Load in parallel
            _n_proc = min(self.n_proc, len(input_paths))  # no more than file_names needed here
            logger.info("Loading {} datasets files using {} parallel processes ...".format(len(input_paths), _n_proc))
            with Pool(processes=_n_proc) as pool:
                all_df = pd.concat(pool.map(PMUData.load_single, input_paths))
        else:  # read 1 file at a time
            all_df = pd.concat(PMUData.load_single(path) for path in input_paths)

        return all_df

    @staticmethod
    def load_single(filepath):
        df = PMUData.read_data(filepath)
        #df = PMUData.select_columns(df)
        num_nan = df.isna().sum().sum()
        if num_nan > 0:
            logger.warning("{} nan values in {} will be replaced by 0".format(num_nan, filepath))
            df = df.fillna(0)

        return df

    @staticmethod
    def read_data(filepath):
        """Reads a single .csv, which typically contains a day of datasets of various weld sessions.
        """
        df = pd.read_csv(filepath)
        return df



data_factory = {'weld': WeldData,
                'hdd': HDD_data,
                'tsra': TSRegressionArchive,
                'semicond': SemicondTraceData,
                'pmu': PMUData}
