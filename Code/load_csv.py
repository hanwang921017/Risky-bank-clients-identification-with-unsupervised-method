import pandas as pd
import numpy as np
from numpy import linalg as LA

def _convert_to_number (s):
    '''
     https://stackoverflow.com/questions/31701991/string-of-text-to-unique-integer-method
    '''

    if not isinstance(s, str):
        if np.isnan(s):
            return s 
        else:
            s = ''
    if isinstance(s, int):
        return s
    return int.from_bytes(s.encode(), 'little')


def _get_two_cols_angle(col1, col2):
    angle =  col1.dot(col2) / (LA.norm(col1)* LA.norm(col2))
    return angle

def _convert_from_number (n):
    '''
     https://stackoverflow.com/questions/31701991/string-of-text-to-unique-integer-method
    '''
    return n.to_bytes(math.ceil(n.bit_length() // 8), 'little').decode()


def load_from_csv(file_name='T8_sample_1000.csv'):
    return pd.read_csv(file_name)


def filled_nan(df, values=-2):
    '''
    fill nan with mean of col
    https://stackoverflow.com/questions/18689823/pandas-dataframe-replace-nan-values-with-average-of-columns
    '''
    df.update(df.fillna(value=values))
    return df, values
    

def prune_null_feature(df, threshold, excludes=[]):
    '''
    prune features that have too many null instances
    if (# of not null)/(# of col) < threshold, remove this feature off
    '''

    instance_size = len(df.values)      
    percentage_df = df.count()/instance_size
    for k,v in percentage_df.items():
        if k not in excludes and  v < threshold:
            df = df.drop(k, axis=1)
    return df


def divide_df(df, prefix_list=['TRD_AL_','TRD_AM_','TRD_TT_']):
    '''
    choose one type
    '''
    dfs = []
    for prefix in prefix_list:
        selector = []
        for col in df.columns:
            if col.startswith(prefix):
                selector.append(col)
            else:
                is_append = True
                for i in prefix_list:
                    if col.startswith(i):
                        is_append=False
                if is_append:
                    selector.append(col)

        dfs.append((prefix, df[selector]))
    
    return dfs


def convert_to_int(df):
    '''
    https://stackoverflow.com/questions/21720022/find-all-columns-of-dataframe-in-pandas-whose-type-is-float-or-a-particular-typ
    '''
    str_df = df.loc[:, df.dtypes == object]
    update_df = str_df.applymap(_convert_to_number)
    df.update(update_df)

    return df


def prune_most_same_value_feature(df, threshold):
    '''
    https://stackoverflow.com/questions/15138973/how-to-get-the-number-of-the-most-frequent-value-in-a-column
    if (# of most frequency value)/(# of col) > threshold, remove this feature off
    '''
    instance_size = len(df.values)
    for k,value in df.items():
        if value.value_counts().max()/instance_size > threshold:
            df = df.drop(k, axis=1)
    return df


def normalization(df, func=lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x) + 0.000000000001)):
    '''
    https://stackoverflow.com/questions/12525722/normalize-data-in-pandas
    '''

    update_df =df[df.columns.difference(['CUST_OID'])]
    df.update(update_df.apply(func))

    return df

def prune_by_similar_cols(df, threshold):

    update_cols = df.columns.difference(['CUST_OID'])
    update_df =df[update_cols]
    for col1 in update_cols:
        for col2 in update_cols:
            if col1 < col2:
                f = col1 
                s = col2
            elif col2 > col1:
                f = col2
                s = col1
            else:
                continue
            similarity = _get_two_cols_angle(update_df[f],update_df[s])
            if similarity > threshold and f in df.columns:
                df = df.drop( columns=f)

    return df
                    


if __name__ == '__main__':
    # here is a exmple
    df = load_from_csv('T8_sample_1000.csv')

    # for prefix, df in dfs:
    #     # convert str to int
    #     df = convert_to_int(df)

    #     # remove features in which there are too many nulls 
    #     df = prune_null_feature(df, 0.5)

    #     # fill nan with mean
    #     df, values = filled_nan(df)

    #     # prune_most_same_value_feature like col: [1,1,1,1,1]
    #     # df = prune_most_same_value_feature(df,0.95)
    #     # normalization with tanh
    #     df = normalization(df)
    #     df = normalization(df,func=lambda x: np.tanh(x.astype(float)))

    #     # prune_similar_cols like [1,2,3] and [2,4,6]
    #     # df = prune_by_similar_cols(df,0.95)
    #     df.to_csv(prefix+'.csv', index=False)
    #     values.to_csv(prefix+'_nan_value.csv', index=False)
    df = convert_to_int(df)
    df, values = filled_nan(df)
    df = normalization(df)
    df = normalization(df,func=lambda x: np.tanh(x.astype(float)))
    df.to_csv('result.csv', index=False)


    



