import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from pandas_profiling import ProfileReport
from pandas.api.types import infer_dtype
from datetime import datetime

CATEGORICAL_PERCENT_THRESOLD = 5
COLUMNS_INFOS=["Name", "NaN", "Null", "Type", "Inferred Type", "Unique Values", "Unique Values (%)", "ML Type", "Count"]
COL_MLTYPE_IGNORE = "Ignore"
COL_MLTYPE_NUMERIC = "Numeric"
COL_MLTYPE_CATEGORICAL = "Categorical"
COL_MLTYPE_Y = "Y"
COL_MLTYPE_TEXT = "Text"

# *********************************************************
def get_col_dtype(col):
  """
  Infer datatype of a pandas column, process only if the column dtype is object. 
  input:   col: a pandas Series representing a df column. 
  """
  if col.dtype == "object":
      try: # try datetime
          col_new = pd.to_datetime(col.dropna().unique())
          return "date" #col_new.dtype
      except:
          try: # try numeric
              col_new = pd.to_numeric(col.dropna().unique())
              return col_new.dtype
          except:
              try: # try time
                  col_new = pd.to_timedelta(col.dropna().unique())
                  return "time" #col_new.dtype
              except:
                  return infer_dtype(col)
  else:
      return infer_dtype(col)

# *********************************************************
def refresh_columns_infos(self, thresold, dataset):
  """
  Method/Function which refresh the columns infos Dataframe
    thresold: numeric (in %) set the thresold between Categorical vs text ratio
    dataset: dataset to investigate/profile
    returns a DataFrame with all the columns informations
  """
  columninfos = pd.DataFrame(columns=COLUMNS_INFOS)
  # Fill the ML column types (Numerical, Categorical or Text)
  categorical_thresold = dataset.shape[0] * thresold / 100
  for col in dataset.columns:
    # Calculate the Nb of value occurence in the columns
    nbOccurrence = len(dataset[col].value_counts())
    # Count the number of nulls
    nbnulls = dataset[col].isnull().sum()
    # Count the NaN
    nbnan = dataset[col].isna().sum()
    # set the ML type
    if (dataset[col].dtypes == "O"):
      if (nbOccurrence >= categorical_thresold):
          mlcolumntype = COL_MLTYPE_TEXT # Text variable if Nb occurence > Thresold (%)
      else:
          mlcolumntype = COL_MLTYPE_CATEGORICAL
    else:
      mlcolumntype = COL_MLTYPE_NUMERIC

    # Create a new Column info
    columninfos["Column"] = columninfos["Name"]
    columninfos = columninfos.append({"Name":col, 
                                      "NaN":nbnan, 
                                      "Null":nbnulls, 
                                      "Type":dataset[col].dtypes, 
                                      "Inferred Type": get_col_dtype(dataset[col]),
                                      "Unique Values":nbOccurrence,
                                      "Unique Values (%)":'{:.2f}'.format(nbOccurrence / dataset.shape[0] * 100),
                                      "ML Type": mlcolumntype,
                                      "Count": dataset.shape[0]}, 
                                      ignore_index=True)
  columninfos = columninfos.set_index(["Name"])
  ds_describe = dataset.describe(include='all').transpose()
  columninfos = columninfos.join(ds_describe)
  try:
    columninfos = columninfos.drop(['count', 'unique'], axis=1)
  except:
    pass
    
  return columninfos

# *********************************************************
def prep_dataset(self, scaler=None):
  """
  Prepare the data before modelisation
    scaler: sklearn scaler engine (must have been initialized before), if None no scaling
  """
  dataset_prep = self.dataset.copy()

  # Drop ignore & Text columns
  ds_ignore = self.columninfos[self.columninfos['ML Type'] == COL_MLTYPE_IGNORE]
  ds_text = self.columninfos[self.columninfos['ML Type'] == COL_MLTYPE_TEXT]
  ds_col_to_remove = pd.concat([ds_ignore, ds_text])
  dataset_prep.drop(ds_col_to_remove.index.to_numpy(), axis=1, inplace=True)

  # Manage Categorical / One-Hot
  ds_categorical = self.columninfos[self.columninfos['ML Type'] == COL_MLTYPE_CATEGORICAL]
  categorical_cols = ds_categorical.index.to_numpy()
  onehot = pd.get_dummies(dataset_prep[categorical_cols], prefix=categorical_cols)
  dataset_prep = pd.concat([onehot, dataset_prep], axis=1)
  dataset_prep.drop(categorical_cols, axis=1, inplace=True)

  # Manage NaN
  ds_numeric = self.columninfos[self.columninfos['ML Type'] == COL_MLTYPE_NUMERIC]
  for col in ds_numeric.index.to_numpy():
    dataset_prep[col] = dataset_prep[col].fillna(0)

  # scale dataset if requested
  if (scaler != None):
    names = dataset_prep.columns
    scaled_data = scaler.fit_transform(dataset_prep)
    dataset_prep = pd.DataFrame(scaled_data, columns=names)

  return dataset_prep

# *********************************************************
def get_data_split(self, dataset, label_column_name, test_size=0.20):
    """
    split dataset in a dataset with label + feature in it
      dataset: dataset to split
      test_size: proportion of the dataset to include in the train split
      label_column_name: Column label name
      returns: X_train, X_test, y_train, y_test
    """
    y = dataset[label_column_name]
    X = dataset.copy()
    X.drop(label_column_name, axis=1, inplace=True)
    return train_test_split(X, y, test_size=test_size)

# *********************************************************
class DatasetManager():
    def __init__(self):
        self.dataset = None 
        self.columninfos = None
        self.scaler = StandardScaler()

    # Returns the columns informations
    def get_columnsinfos(self):
        return self.columninfos

    # Returns the dataset as DataFrame
    def get_dataset(self):
      return self.dataset

    # Private method which refresh the columns infos Dataframe
    __refresh_columns_infos = refresh_columns_infos

    # data preparation function
    prep = prep_dataset

    # Display the Pandas Profiling Report
    def profile(self, minimal=True):
      profile = ProfileReport(
                      self.dataset, 
                      minimal=minimal, 
                      title="Dataset Profiling Report"
                )
      profile.to_notebook_iframe()

    # Set/Force ML Column Types
    def __set_colmltype(self, columnnames, status):
      for colname in columnnames:
        self.columninfos.loc[colname]["ML Type"] = status
    def set_colmltype_categorical(self, columnnames):
      self.__set_colmltype(columnnames, COL_MLTYPE_CATEGORICAL)
    def set_colmltype_numeric(self, columnnames):
      self.__set_colmltype(columnnames, COL_MLTYPE_NUMERIC)
    def set_colmltype_text(self, columnnames):
      self.__set_colmltype(columnnames, COL_MLTYPE_TEXT)
    def set_colmltype_ignore(self, columnnames):
      self.__set_colmltype(columnnames, COL_MLTYPE_IGNORE)
    def set_colmltype_Y(self, columnnames):
      self.__set_colmltype(columnnames, COL_MLTYPE_Y)

    # Returns specific column info
    def get_column_info(self, columnname):
      return self.columninfos.loc[columnname]

    # Load the data file and initialize/populate internal variables/infos
    def load_csv(self, filepath, cat_text_thresold=CATEGORICAL_PERCENT_THRESOLD):
        self.dataset = pd.read_csv(filepath)
        self.columninfos = self.__refresh_columns_infos(cat_text_thresold, self.dataset)

    # Load the data file and initialize/populate internal variables/infos
    def load_dataframe(self, dataset, cat_text_thresold=CATEGORICAL_PERCENT_THRESOLD):
        self.dataset = dataset
        self.reset(cat_text_thresold)

    # Reset the infos dataframe from the current dataset data
    def reset(self, cat_text_thresold=CATEGORICAL_PERCENT_THRESOLD):
        if not(self.dataset is None):
          self.columninfos = self.__refresh_columns_infos(cat_text_thresold, self.dataset)

    def change_column_name(self, oldname, newname):
      for i in range(len(self.columninfos)):
        if (self.columninfos["Column"][i] == oldname):
          self.columninfos["Column"][i] = newname

    def manage_date_column(self, col, 
                           date_format="%Y/%m/%d", 
                           default_date="01/01/1901 00:00:00", default_date_format='%d/%m/%Y %H:%M:%S',
                           dropcolumn=True):
      self.dataset[col] = pd.to_datetime(self.dataset[col], errors='coerce', format=date_format)
      def_default_date = lambda mydate: datetime.strptime(default_date, default_date_format) if pd.isnull(mydate) else mydate
      self.dataset[col] = self.dataset[col].apply(def_default_date)
      self.dataset[col + "_weekday"] = self.dataset[col].dt.day_name()
      self.dataset[col + "_day"] = self.dataset[col].dt.day
      self.dataset[col + "_month"] = self.dataset[col].dt.month
      self.dataset[col + "_year"] = self.dataset[col].dt.year
      if (dropcolumn):
        self.dataset = self.dataset.drop([col], axis=1)

    # split dataset in a dataset with label + feature in it
    get_data_split = get_data_split
    
    def randomoversample(self, X_train, y_train, sampling_strategy='minority'):
        os = RandomOverSampler(sampling_strategy=sampling_strategy)
        # Convert to numpy (mandatory before)
        x_np, y_np = X_train.to_numpy(), y_train.to_numpy()
        # Oversample
        x_np, y_np = os.fit_resample(x_np, y_np)
        # Convert back to pandas
        x_df = pd.DataFrame(x_np, columns=X_train.columns)
        y_df = pd.Series(y_np, name=y_train.name)
        return x_df, y_df