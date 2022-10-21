## Pandas 101

This notebook contains frequently used Pandas commands for data preprocessing and manipulation.

- [Miscellaneous](#Miscellaneous)
- [Data Input](#Data-Input)
- [High Level Information](#High-Level-Information)
- [Filtering & Indexing](#Filtering--Indexing)
- [Grouping & Aggregation](#Grouping--Aggregation)
- [Merging](#Merging)
- [Modification](#Modification)
- [Handling Missing Data](#Handling-Missing-Data)
- [Handling Time Series Data](#Handling-Time-Series-Data)

### Miscellaneous

- Install Pandas:
  - pip install pandas (add **'!'** at the begining in jupyter notebook)
  - pip install pandas=='version id' (a specific version)
- Version:
  - print(pandas.\_\_version\_\_)
  - pip install --upgrade pandas
- Setting
  - pandas.set_option('display.max_row',None)
- Import Pandas
  - import pandas as pd

### Data Input

    Depending on the data sources, pandas offer many options, the most frequently used method is 'read_csv', 'read_excel', 'read_html'.

- read_csv()
  - sep: '\t' --> tab; '\s+' --> white space; ',' --> comma, etc.
  - header: None --> no header; 0 --> first row as header
  - names: **List**, set column names
  - index_col: set index
  - dtype: **Dictionary**, set data type for each column
  - skiprows: skip rows from the begining
- read_excel():
  - sheet_name: specify the excel sheet

### High Level Information

    Commands used to get a feeling of your data.

- df.columns.tolist(): get column names
- df.index: get indices
- df.shape: # of row by # of column
- df.size: # of cell
- df.info(): summary of dataframe
- df.describe(): return statistics of numeric columns
- df.dtypes: return data types of each column
- df.corr: get correlation coefficients

### Filtering & Indexing

    Commands used to select a subgroup of the dataframe.

- Indexing:
  - df.iloc(): using integer location, exclusive when slicing, can use **_np.r\__** to represent multiple slice
  - df.loc(): using label, inclusive when slicing
- Filtering:
  - ['columnA']
  - boolean indexing
  - df.query()
  - df.select_dtypes()

### Grouping & Aggregation

    Aggregation is done by using the 'groupby' method and 'agg' method in combination.
  - grouby(['columnA','columnB'])
  - agg(['mean','median','sum','prod','std','var','max','min','count','describe','nunique',])

### Merging

  - pd.concat
  - df.append
  - df.merge
  - df.join

### Modification
  - Modifying column name: 
    - df.columns = ['columnA','columnB',...]
    - df.rename(columns={})
  - Modifying index:
    - df.set_index('index_column',inplace = True): use a column as the nex index
    - df.reset_index(inplace = True, , drop = True): reset index to defalt
  - Modifying a column:
    - df.apply(lambda, axis=1): can be used for multiple columns
    - df.map(lambda): only for Series
  - Modifying all cells:
    - df.applymap(lambda)

### Handling Missing Data
  - Checking missing values:
    - pd.isnull(df) or pd.isna(df)
    - df.isnull() or df.isna()
  - Filling missing values:
    - df.fillna()
    - sklearn imputation
    
### Handling Time Series Data
  - Converting to datetime:
    - pd.to_datetime()
    - df['DateTime'].dt.year
    - df['DateTime'].dt.month
    - df['DateTime'].dt.day
    - df['DateTime'].dt.week
  - Aggregating:
    - df.resample()
    
