# Risky-bank-clients-identification-with-unsupervised-method
This work is a project cooperating with ATB bank. It is to identify the risky clients based on the clients’ transaction history. Risky clients represent the clients who are going to default on a loan or debt. Our team is required to use an unsupervised method to do this work. First of all, we do the feature engineering on the data. For example, we filled the null with the average of their columns and then we normalize it to eliminate the features’ difference in scale. And then we use PCA to reduce the dimensionality of the data. It is reduced from 1200 features to around 200 features. This can improve the efficiency a lot. Next, we select two kinds of popular unsupervised models including the clustering models like K-means and probabilistic models like GMM. Cross-validation is used to tune the parameters in the training process. Lastly, Silhouette coefficient is used to evaluate clustering models. While log-likelihood is used to evaluate the probabilistic models. The best model we find is the GMM model. It can identify the risky proportion around 28%.
# Using machine learning to detect anomaly in transactions



```bash
$ virtualenv venv --python=python3
$ source venv/bin/activate
$ pip install -r requirements.txt

```



## How to use load_csv.py



[prune_by_similar_cols](#prune_by_similar_cols)

[normalization](#normalization)

[prune_most_same_value_feature](#prune_most_same_value_feature)

[convert_to_int](#convert_to_int)

[divide_df](#divide_df)

[prune_null_feature](#prune_null_feature)

[filled_nan](#filled_nan)

[load_from_csv](#load_from_csv)

### load_from_csv
(file_name)

```python
df = load_from_csv(file_name)

		CUST_OID POSTAL_CODE    ...      INQ_AM_30  INQ_AM_31
0          1         XXX    ...              0        NaN
1          2         T3K    ...              1       16.0
2          3         T2X    ...              3        6.0
3          4         T3H    ...              2        1.0
4          5         T8N    ...              2        0.0
5          6         T2Z    ...              2        2.0
6          7         T8V    ...              2       14.0
7          8         T0H    ...              2        2.0
8          9         T9W    ...              2       14.0
9         10         T5H    ...              1       22.0
10        11         T0J    ...              7        2.0
11        12         T0H    ...              1       13.0
12        13         T0L    ...              0        NaN
13        14         T1V    ...              3        8.0
14        15         T0J    ...              1       10.0
15        16         T0B    ...              0       32.0
16        17         T8S    ...             10        3.0
17        18         T0K    ...              0        NaN
18        19         T1A    ...              0       27.0

[19 rows x 16 columns]
```

### convert_to_int
(df)

```python
df = convert_to_int(df)

	   CUST_OID POSTAL_CODE    ...      INQ_AM_30  INQ_AM_31
0          1     5789784    ...              0        NaN
1          2     4928340    ...              1       16.0
2          3     5780052    ...              3        6.0
3          4     4731732    ...              2        1.0
4          5     5126228    ...              2        0.0
5          6     5911124    ...              2        2.0
6          7     5650516    ...              2       14.0
7          8     4730964    ...              2        2.0
8          9     5716308    ...              2       14.0
9         10     4732244    ...              1       22.0
10        11     4862036    ...              7        2.0
11        12     4730964    ...              1       13.0
12        13     4993108    ...              0        NaN
13        14     5648724    ...              3        8.0
14        15     4862036    ...              1       10.0
15        16     4337748    ...              0       32.0
16        17     5453908    ...             10        3.0
17        18     4927572    ...              0        NaN
18        19     4272468    ...              0       27.0

[19 rows x 16 columns]
```



### filled_nan
(df, values=None)

```python
df = convert_to_int(df)
df = filled_nan_with_mean(df)

	CUST_OID POSTAL_CODE    ...      INQ_AM_30  INQ_AM_31
0          1     5789784    ...              0      10.75
1          2     4928340    ...              1      16.00
2          3     5780052    ...              3       6.00
3          4     4731732    ...              2       1.00
4          5     5126228    ...              2       0.00
5          6     5911124    ...              2       2.00
6          7     5650516    ...              2      14.00
7          8     4730964    ...              2       2.00
8          9     5716308    ...              2      14.00
9         10     4732244    ...              1      22.00
10        11     4862036    ...              7       2.00
11        12     4730964    ...              1      13.00
12        13     4993108    ...              0      10.75
13        14     5648724    ...              3       8.00
14        15     4862036    ...              1      10.00
15        16     4337748    ...              0      32.00
16        17     5453908    ...             10       3.00
17        18     4927572    ...              0      10.75
18        19     4272468    ...              0      27.00

[19 rows x 16 columns]
```

### prune_null_feature
(df, threshold, excludes=[])

```python
df = prune_null_feature(df,0.8)

    CUST_OID POSTAL_CODE    ...      INQ_AM_30  INQ_AM_31
0          1         XXX    ...              0        NaN
1          2         T3K    ...              1       16.0
2          3         T2X    ...              3        6.0
3          4         T3H    ...              2        1.0
4          5         T8N    ...              2        0.0
5          6         T2Z    ...              2        2.0
6          7         T8V    ...              2       14.0
7          8         T0H    ...              2        2.0
8          9         T9W    ...              2       14.0
9         10         T5H    ...              1       22.0
10        11         T0J    ...              7        2.0
11        12         T0H    ...              1       13.0
12        13         T0L    ...              0        NaN
13        14         T1V    ...              3        8.0
14        15         T0J    ...              1       10.0
15        16         T0B    ...              0       32.0
16        17         T8S    ...             10        3.0
17        18         T0K    ...              0        NaN
18        19         T1A    ...              0       27.0
[19 rows x 15 columns]
```



### divide_df
(df, prefix_list=['TRD_AL', 'TRD_AM', 'TRD_TT'])

```python
# return dfs contains a list of (prefix_name, df)
dfs = divide_df(df)
for prefix,adf in dfs:
    df.to_csv(prefix+'.csv')
    
```



### prune_most_same_value_feature
(df, threshold)

```python
df = convert_to_int(df)
df = filled_nan_with_mean(df)
df = prune_most_same_value_feature(df,0.8)

		CUST_OID POSTAL_CODE  SUBJECT_AGE    ...      INQ_AM_29  INQ_AM_30  INQ_AM_31
0          1     5789784           59    ...              0          0      10.75
1          2     4928340           51    ...              0          1      16.00
2          3     5780052           49    ...              2          3       6.00
3          4     4731732           42    ...              2          2       1.00
4          5     5126228           30    ...              2          2       0.00
5          6     5911124           32    ...              1          2       2.00
6          7     5650516           34    ...              0          2      14.00
7          8     4730964           31    ...              1          2       2.00
8          9     5716308           26    ...              0          2      14.00
9         10     4732244           44    ...              0          1      22.00
10        11     4862036           46    ...              6          7       2.00
11        12     4730964           32    ...              0          1      13.00
12        13     4993108           92    ...              0          0      10.75
13        14     5648724           49    ...              1          3       8.00
14        15     4862036           63    ...              1          1      10.00
15        16     4337748           36    ...              0          0      32.00
16        17     5453908           26    ...              4         10       3.00
17        18     4927572           84    ...              0          0      10.75
18        19     4272468           47    ...              0          0      27.00

[19 rows x 13 columns]
```



### prune_by_similar_cols
(df, threshold)



```python
df = convert_to_int(df)
df = filled_nan_with_mean(df)
df = prune_by_similar_cols(df,0.9)

 CUST_OID  SUBJECT_AGE    ...      INQ_AM_30  INQ_AM_31
0          1           59    ...              0      10.75
1          2           51    ...              1      16.00
2          3           49    ...              3       6.00
3          4           42    ...              2       1.00
4          5           30    ...              2       0.00
5          6           32    ...              2       2.00
6          7           34    ...              2      14.00
7          8           31    ...              2       2.00
8          9           26    ...              2      14.00
9         10           44    ...              1      22.00
10        11           46    ...              7       2.00
11        12           32    ...              1      13.00
12        13           92    ...              0      10.75
13        14           49    ...              3       8.00
14        15           63    ...              1      10.00
15        16           36    ...              0      32.00
16        17           26    ...             10       3.00
17        18           84    ...              0      10.75
18        19           47    ...              0      27.00

[19 rows x 6 columns]

```



### normalization
(df, func=...)

```python
df = convert_to_int(df)
df = filled_nan_with_mean(df)
df = prune_most_same_value_feature(df,0.8)
df = normalization(df)	
    CUST_OID POSTAL_CODE  SUBJECT_AGE    ...      INQ_AM_29  INQ_AM_30  INQ_AM_31
0          1     5789784           59    ...              0          0      10.75
1          2     4928340           51    ...              0          1      16.00
2          3     5780052           49    ...              2          3       6.00
3          4     4731732           42    ...              2          2       1.00
4          5     5126228           30    ...              2          2       0.00
5          6     5911124           32    ...              1          2       2.00
6          7     5650516           34    ...              0          2      14.00
7          8     4730964           31    ...              1          2       2.00
8          9     5716308           26    ...              0          2      14.00
9         10     4732244           44    ...              0          1      22.00
10        11     4862036           46    ...              6          7       2.00
11        12     4730964           32    ...              0          1      13.00
12        13     4993108           92    ...              0          0      10.75
13        14     5648724           49    ...              1          3       8.00
14        15     4862036           63    ...              1          1      10.00
15        16     4337748           36    ...              0          0      32.00
16        17     5453908           26    ...              4         10       3.00
17        18     4927572           84    ...              0          0      10.75
18        19     4272468           47    ...              0          0      27.00

[19 rows x 13 columns]
```

