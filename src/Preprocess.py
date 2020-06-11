import pandas as pd
import ModelConfig
import os
import numpy as np
from api_logger import logger

class Preprocess:

    def __init__(self):
        self.input = pd.read_csv(os.path.join('../input', self.filename))
        self.training_schema = ModelConfig.training_schema
        self.training_data_columns = list(self.training_schema.keys())
        self.training_impute = ModelConfig.training_impute
        self.model_features =  ModelConfig.model_features
        self.cat_cols = ModelConfig.cat_cols
        logger.info(f'Preprocessing starts...')

    def data_quality_check(self):

        def _2str(x):
            if x == x:
                try:
                    x = str(x)
                    x = x.lower()
                except:
                    x = np.nan
            return x

        def _2int(x):
            try:
                x = int(x)
            except:
                x = np.nan
            return x

        def _2float(x):
            try:
                x = float(x)
            except:
                raise ValueError(f'{col_name} contains {x} , expected only float')
            return x

        col_required = [col for col in self.training_data_columns if col not in self.input.columns.tolist()]

        # If all required columns are not found
        if len(col_required) > 0:
            raise ValueError(
                f'Following columns was not passed {col_required}.\n Note: Columns name are case sensitive. ')

        self.input = self.input[self.training_data_columns]

        # id can not be blank
        if self.input['sku'].isna().sum() / self.input.shape[0] > 0:
            raise ValueError('sku is an identifier and can not be null.')

        # id can not be float
        if self.input.sku.dtype != 'int':
            raise ValueError(f'sku should be int found {self.input.sku.dtype}')

        logger.info(f'Checking for schema compatibility...')

        # Check for datatype of each columns
        for col_name, data_type in self.training_schema.items():
            if data_type == 'str':
                self.input[col_name] = self.input[col_name].apply(lambda x: _2str(x))
            elif data_type == 'int':
                self.input[col_name] = self.input[col_name].apply(lambda x: _2int(x))
            elif data_type == 'float':
                self.input[col_name] = self.input[col_name].apply(lambda x: _2float(x))
            else:
                raise ValueError('Unexpected dtype %s specified for columns %s' % (data_type, col_name))


        # converting negative values into null values
        for col in ['national_inv','perf_6_month_avg', 'perf_12_month_avg']:
            self.input[col] = self.input[col].mask(self.input[col] < 0)

        # Check for null values
        na_count = (self.input.isna().sum().sum() * 100/ (self.input.shape[0] * self.input.shape[1])).astype('int')
        if na_count > 30:
            raise ValueError(f'{na_count}% null values. Insufficient data for prediction.')

        logger.info(f'{na_count}% null values. Null values details below \n {self.input.isna().sum()/self.input.shape[0]}')

        # Check for specific value in categorical variables
        for col in self.cat_cols:
            val = [val_ for val_ in self.input[col].unique().tolist() if val_ not in ['yes', 'no']]

            if len(val) > 0:
                raise ValueError(f'{col} has value {val} ; expected only yes,no values ')

