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
from sklearn import preprocessing
from datetime import  datetime

import matplotlib.pyplot as plt
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
        self.types = None
        self.labelEncoder = {}

    def normalize(self, df):
        """
        Args:
            df: input dataframe
        Returns:
            df: normalized dataframe
        """
        # logger.info(" Normalize : shape  {} \n columns {}", .format(df.shape,  df.head()))
        self.types = df.dtypes
        # if any feature in Numeric then done label-encoding;
        charcolumns =  df.columns[self.types == "object"]
        for cols in charcolumns:
            self.labelEncoder[cols] = preprocessing.LabelEncoder().fit(df[cols])
            df[cols] = self.labelEncoder[cols].transform(df[cols])

        if self.norm_type == "standardization":
            if self.mean is None:
                self.mean = df.mean()
                self.std = df.std()
            return (df - self.mean) / (self.std + np.finfo(float).eps)

        elif self.norm_type == "minmax":
            if self.max_val is None:
                self.max_val = df.max()
                self.min_val = df.min()
                logger.info("normalize: \nmax {} \n min{} ".format(self.max_val, self.min_val))
            return (df - self.min_val) / (self.max_val - self.min_val + np.finfo(float).eps)

        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            self.groupedidx = grouped.indices
            self.groupedmean = grouped.transform('mean')
            self.groupedstddev = grouped.transform('std')
            return (df - self.groupedmean) / self.groupedstddev

        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            self.groupedidx  = grouped.indices
            self.groupedmin_vals = grouped.transform('min')
            self.groupedmax_vals = grouped.transform('max')
            return (df - self.groupedmin_vals) / (self.groupedmax_vals - self.groupedmin_vals + np.finfo(float).eps)

        else:
            raise (NameError(f'Normalize method "{self.norm_type}" not implemented'))

    def inverse_normalize(self, df:pd.DataFrame=None):
        """reverses the transformation done with the method that was used normalize function above.

        Args:
            df (_type_): dataFrame with the same names as that was given earlier
        """
        dfout = pd.DataFrame()
        # inverse normalize the character columns once done with reversing of the minmax

        if self.norm_type == "standardization":
            assert set(self.mean.index).issuperset(set(df.columns))
            means = self.means[df.columns]
            stds = self.std[df.columns]
            dfout =  pd.DataFrame(df*(stds+ np.finfo(float).eps) +  means)

        elif self.norm_type == "minmax":
            assert set(self.max_val.index).issuperset(set(df.columns))
            minvals = self.min_val[df.columns]
            maxvals = self.max_val[df.columns]
            dfout = pd.DataFrame((maxvals - minvals + np.finfo(float).eps)*df + minvals)

        elif self.norm_type == "per_sample_std":
            logger.info( "---------------- YTD ------------------------")
            idx  = df.index
            means = self.groupedmean[idx]
            stds = self.groupedstddev[idx]
            dfout = pd.DataFrame(df*stds + means)

        elif self.norm_type == "per_sample_minmax":
            idx = df.index
            min_vals = self.groupedmin_vals[idx]
            max_vals = self.groupedmax_vals[idx]
            dfout = pd.DataFrame(df * (max_vals - min_vals + np.finfo(float).eps) - min_vals)

        else:
            raise (NameError(f'Normalize method "{self.norm_type}" not implemented'))
            return None
        
        dfout.columns = df.columns
        charcolumns = dfout.columns[dfout.columns.dtypes == "object"]
        for cols in charcolumns:
            dfout[cols] = self.labelEncoder.inverse_transform(dfout[cols])
        
        return dfout


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
                data = load_data.load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                                     replace_missing_vals_with='NaN')
                if isinstance(data, tuple):
                    df, labels = data
                else:
                    df = data
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

 
## create a class of surveillance data
class WasteWaterData(BaseData):
    """
    Dataset class for Machine dataset.
    Attributes:
        all_df: dataframe indexed by ID, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.

    """
    def __init__(self, root_dir, file_list=None, pattern=None, n_proc=1, limit_size=None,
            navalue = None, config=None):
        
        self.config = config
        self.wideData =None
        self.wideDataFinal = None
        self.feature_df = None
        self.set_num_processes(n_proc=n_proc)
        self.Normalizer = None
        all_df = None

        try:
            # print(" ---- init 0: ", root_dir )
            all_df = self.load_all(root_dir, file_list=file_list, pattern=pattern)

            wwdata, self.demodata = self.processData(all_df)

        except Exception as e:
            print(" __init__ 1:")
            print(all_df.head())

            raise 
        
        try:
            self.all_df = self.cleanData(wwdata)
            self.all_df = self.all_df.set_index('days')
            self.all_IDs = self.all_df.index.unique()  # all sample (session) IDs
            self.navalue = navalue
            self.max_seq_len = 66
            if limit_size is not None:
                if limit_size > 1:
                    limit_size = int(limit_size)
                else:  # interpret as proportion if in (0, 1]
                    limit_size = int(limit_size * len(self.all_IDs))
                self.all_IDs = self.all_IDs[:limit_size]
                all_df = all_df.loc[self.all_IDs]
        
        except Exception as e:
            print(" __init__ 2:")
            print(wwdata.head())
            raise

        """
        sets variable that are features, uses regex feature match to create the list
        of features that are needed to generate the sequence.

        feature_df: contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
        """
        feature_names = ['geoLat','geoLong','phureg', 'sewershedPop',
                        'mQ', 'mN1', 'mN2', 'mBiomarker']
        
        self.feature_df = (self.all_df.loc[self.all_df.phureg.isin(['Central East','Toronto','Central West']).values,feature_names]
                        .reset_index().sort_values(['days','phureg','geoLat','geoLong']).set_index('days'))
        self.feature_names = feature_names

        # replacing the missing with -1 in mQ, mN1, mN2, and mBiomarker
        self.feature_df.loc[:,['mQ', 'mN1', 'mN2', 'mBiomarker']] = self.feature_df.loc[:,['mQ', 'mN1', 'mN2', 'mBiomarker']].fillna(np.nan, inplace=True)
        logger.info("data shape is {}\n names: {}\n".format(self.feature_df.shape,self.feature_names))

    def load_all(self, root_dir, file_list=None, pattern=None):

        # read in MECP data
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_dir,'*')) # list of all paths
        else:
            data_paths = [os.path.join(root_dir, p) for p in file_list]
            
        if pattern is None:
            # by default evaluate on
            selected_paths = data_paths
        else:
            selected_paths = list(filter(lambda x: re.search(pattern, x), data_paths))

        input_paths = [p for p in selected_paths if os.path.isfile(p) and p.endswith('.csv')]
        print(" load_all .... ", data_paths,input_paths)

        if len(input_paths) == 0:
            raise Exception(f"No .csv files found using pattern: {pattern}")
        
        if self.n_proc >1:
            #do this....if you really have big data
            logger.info(" ----------------------------YTD ---------------------\n")
            all_df = pd.concat(WasteWaterData.load_single(path) for path in input_paths)
        else:
            logger.info(" ----------------- concatenating-----------------------------")
            all_df = pd.concat(WasteWaterData.load_single(path) for path in input_paths)

        return all_df

    def set_data(self, phureg=None, vars= None, scale=True):
        """ creates the wide data from the dataset for slice based on the 'phureg' and 'vars',
        output is data frame which is stored in parameter 'X' with column names having the latitude
        and longitude information prepended with 'its vars' e.g. var1_<lat>_<long>

        Args:
            phureg (_type_, optional): one of the regions that you want to consider e.g. GTA. Defaults to None.
            vars (_type_, optional): These are the variable obtained from the self.wwdata . Defaults to None.
            normalize (_str_) : if the data has to be normalized before being transformed into wider format

        """
        if scale:
            self.Normalizer = Normalizer('minmax')
            wideDatalist = WasteWaterData.Transform(self.all_df, phureg=phureg, varlist=vars, 
                    normalizer=self.Normalizer)
        else :
            wideDatalist = WasteWaterData.Transform(self.all_df, phureg=phureg, varlist=vars)
        self.wideData = pd.concat(wideDatalist, axis=1, ignore_index=False)
        self.X = self.wideData
    
    def append_data(self, data:pd.DataFrame=None,axis=1):
        """
        This function lets one add another dataset to the already created wide dataset in wide format
        using the index of days, note that indexes can be retrived on wideData
        Note that the object doesn't remember any new data that was added already if you want to 
        add more data then you have to merge the data outside of the function and then call this function 
        to append it to the root "wideData".

        Args:
            data (pd.DataFrame, optional): Data frame with row index same as the row index
            of the wideData. Defaults to None.

        """
        self.X = pd.concat( [self.wideData, data], axis=axis,join='inner')

    def processData(self,data:pd.DataFrame=None):
        """_summary_
        Args:
            data (_type_, optional): _description_. Defaults to None.
        Returns:
            _type_: _description_
        """

        data["date"] = data.sampleDate.astype('datetime64')
        data.rename(columns={"publicHealthRegion":"phureg", "publicHealthUnit":"phu"}, inplace=True)
        logger.info("data shape is {}".format(data.shape))

        self.idxcols = ['phureg','phu','geoLat','geoLong']
        self.origcols = ['mQ', 'mN1', 'mN2', 'mN1N2', 'mBiomarker']
        self.mavars = ['mN1_7dma', 'mN2_7dma',  'mN1N2_7dma','mN1E_7dma', 'nmN1_7dma', 'nmN2_7dma','nmN1N2_7dma', 
              'nmE_7dma',  'nmN1E_7dma', 'qmN1N2', 'nqmN1N2','qmE', 'qmN1E', 'nqmE', 'nqmN1E','nmE', 'nmN1E',
               'mE', 'mN1E']

        self.othervars = ['date','sewershedPop', 'sewershedPop2020', 'siteID', 'nmN1', 'nmN2',
                        'nmN1N2',  'qmN1', 'qmN2', 'qmN1N2',  'nqmN1', 'nqmN2','nqmN1N2']

        self.demovars = ['moh_cbed','moh_cbed_7dma', 'moh_abed', 'moh_cbrd', 'moh_cbrd_7dma',
                        'moh_covidHospitalization', 'moh_covidHosp_7dma',
                        'moh_vaccine_atLeastTwoDoses']

        # keep only required columns;
        dat3 = data[ self.idxcols + self.origcols + self.mavars + self.othervars + self.demovars ].copy()
 
        dat3 = dat3[dat3.date >= datetime.strptime("01/04/2021","%d/%m/%Y")]
        dat3_1 = dat3[self.idxcols + self.demovars].copy()
        dat3 = dat3[ self.idxcols + self.othervars + self.origcols ].copy()

        return dat3,dat3_1

    @staticmethod
    def load_single(filepath):
        print(f'Load_single: {filepath}\n')
        df = pd.read_csv(filepath, low_memory=False)
        # df = WasteWaterDataClass.select_columns(df)
        num_nan = df.isna().sum().sum()
        if num_nan > 0:
            logger.warning("{} nan values in {} will be replaced with 0".format(num_nan,filepath))

        return df

    @staticmethod
    def Transform(df, phureg = 'gta_region', varlist=["nmN1"], index=None, 
                    normalizer:Normalizer = None ):
        """Normalization routine for the data we have and it can be done by phu-region and variable list
        
        Args:
            df (_type_): _description_
            phureg (str, optional): _description_. Defaults to 'gta_region'.
            varlist (list, optional): List of variables to used for normalization. Defaults to ["nmN1"].
            index (_type_, optional): _description_. Defaults to None.
            normalizer (Normalizer, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if index is not None :
            df = df.set_index(index)

        if not isinstance(phureg,list):
            phureg = [phureg]

        df = df[df["phureg"].isin(phureg)].copy()

        #Extract columns to build model
        df = df[["geoLat", "geoLong"]+ varlist]

        if normalizer is not None:
            # print(f'\t Normalizing using {normalizer.norm_type}...\n')
            df[varlist] = normalizer.normalize(df[varlist].copy())

        #Group the data into a multivariate time series grouped by geoLat and geoLong
        df_var = []
        for i,vars in enumerate(varlist):
            df_var.append(df.groupby(["geoLat", "geoLong"])[vars].apply(list).reset_index(drop=False))

        #Extract geoLat and geoLong to make labels for time series, then drop the columns
        names = [ str(a[0]) + '_' + str(a[1]) for a in list(zip(df_var[0]["geoLat"], df_var[0]["geoLong"])) ] 

        # replace the colum names prepended with the latitude and longitude.
        for i in range(len(df_var)):
            df_var[i].drop(columns=["geoLat", "geoLong"], inplace=True)

        #List -> columns format
        for i, vars in enumerate(varlist):
            df_var[i] = pd.DataFrame(df_var[i][vars].tolist(), index=df_var[i].index).transpose()
            df_var[i].columns = [ vars + '_' + name for name in names]
    
        return df_var

    def cleanData(self, df:pd.DataFrame):

        t1 = df[self.origcols].isna().apply(sum,axis=1)
        df3 = df[t1 <=30].sort_values(self.idxcols +['date'])

        df3['days'] = (df3.date - pd.to_datetime('2021-04-01')).dt.days
        
        global_max = max(df3.days)
        global_min = min(df3.days)
        # function to check contiguousness of the days if not add missing line in between
        def checknadd_missingdays(XX :pd.DataFrame = None ):
            assert XX.days.unique().shape[0] == XX.shape[0], "====Days are not unique here===="
            daysdf = pd.DataFrame({'days': range(global_min,global_max)}).reset_index(drop=True)
            XX = daysdf.merge(XX, how = 'left', on='days')
            return XX

        logger.info('adding missing dates in the data')
        df4 = (df3.groupby(self.idxcols)
                    .apply(lambda x: checknadd_missingdays(x.drop(columns=self.idxcols)))
                    .reset_index().drop(columns=['date','level_4'])
                    )

        # impute the sewershedpop if missing from 2020 after backfill and forward fill grouping by 
        #  region/PHU latitude and longitude.    
        logger.info("imputing the seweshed population")
        df4['sewershedPop'] = (df4.groupby(self.idxcols)['sewershedPop']
                                        .apply(lambda x: x.ffill().bfill())
                                        .reset_index(drop=True)
                                        ).values
        df4['sewershedPop2020'] = (df4.groupby(self.idxcols)['sewershedPop2020']
                                        .apply(lambda x: x.ffill().bfill())
                                        .reset_index(drop=True)
                                        ).values
        df4.loc[df4.sewershedPop.isna(),'sewershedPop'] = df4.loc[df4.sewershedPop.isna(),'sewershedPop2020']

        # dropping lat long where there less than 10% non-null.
        print("Dropping Latitude and Longitude with less than 10% non-missing in all columns \n\
                of the original values....")
        _X = (df4[self.idxcols + self.origcols ].groupby(['geoLat','geoLong'])
            .apply(lambda x : x.notnull().mean()).drop(columns=self.idxcols) )
        drop_rows = (_X<.1).all(axis=1)
        drop_lat_long = _X[drop_rows].reset_index().iloc[:,:2]
        

        logger.info('dropping following columns :\n\t {} ...'.format(drop_lat_long))
        df4 = df4[~df4.geoLat.isin(drop_lat_long.geoLat) & ~df4.geoLong.isin(drop_lat_long.geoLong)]

        # print( '--------------restricting length of latitude/Longitude ----------------' )
        df4.geoLat = df4.geoLat.round(4)
        df4.geoLong = df4.geoLong.round(4)
        logger.info( '------------ done ...-------------')
        return df4

    @staticmethod
    def mytsplot(df:pd.DataFrame = None, labels = None, xlabel=None, color='rgmbckyte', ylog=False,
            title="", ylim=None):
        if labels is None :
            labels = df.columns.values
        n = df.shape[1]
        maxvalue = df.max().values.max()
        ymin = max(df.min().values.min(), 8e-6)
        fig = plt.figure(figsize=(20,12),dpi=120)
        ax = fig.add_subplot()
        ax.plot(df, linewidth=1.5 )
        hdl = ax.get_legend_handles_labels()
        if ylim is not None:
            ax.set_ylim(ymin=ylim[0],ymax=ylim[1])
        else:
            ax.set_ylim(ymin=ymin, ymax=maxvalue)
        if ylog:
            ax.set_yscale('log')
        fig.legend(hdl,labels=labels,bbox_to_anchor=[1.2,0.6], loc='center right')
        ax.set_title(title)

        
data_factory = {'weld': WeldData,
                'tsra': TSRegressionArchive,
                'pmu': PMUData,
                'wastewater': WasteWaterData}
