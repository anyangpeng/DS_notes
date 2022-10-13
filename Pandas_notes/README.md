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

### Filtering & Indexing

    Commands used to select a subgroup of the dataframe.

- Indexing:
  - df.iloc(): using integer location, exclusive when slicing, can use **_np.r\__** to represent multiple slice
  - df.loc(): using label, inclusive when slicing
- Filtering:
  - boolean indexing
  - df.query()
  
### Grouping & Aggregation

### Merging

### Modification

### Handling Missing Data

### Handling Time Series Data
