import pandas as pd

def preprocess_static(flats, cat_threshold, path, unique_hadmids):
  static_data = flats.copy()
  static_data.rename(columns={'admissionid':'hadmid'}, inplace=True)
  '''
  This needs to be done on selected admission ids
  '''

  static_data.set_index('hadmid', inplace=True)  ; static_data = static_data.loc[unique_hadmids]

  static_data['gender'].replace({'Man': 1, 'Vrouw': 0}, inplace=True)
  #don't need a transformation for urgency
  cat_features = ['location', 'admissioncount', 'origin', 'admissionyeargroup', 'agegroup', 'weightgroup', 'heightgroup', 'specialty']
  #imputation using a new category
  imputation_dict = {'origin':'no_origin', 'weightgroup':'no_weight_group', 'heightgroup':'no_height_group', 'gender':0.5}
  static_data.fillna(imputation_dict, inplace=True)
  cols_to_drop = ['dateofdeath', 'lengthofstay', 'patientid']
  static_data = static_data.drop(columns = cols_to_drop)
  
  for cat in cat_features:
    too_rare = [value for value, count in static_data[cat].value_counts(dropna=False).iteritems() if count < cat_threshold]
    static_data.loc[static_data[cat].isin(too_rare), cat] = 'misc'
  
  static_data = pd.get_dummies(static_data, columns=cat_features)

  return static_data.reset_index().to_csv(path + 'preprocessed_static.csv', index=False)


def preprocess_labels(labels, labels_vars, path, unique_hadmids):
  
  labels_data = labels[labels_vars]
  labels_data.rename(columns={'admissionid':'hadmid'}, inplace=True)
  labels_data.set_index('hadmid', inplace=True) ; labels_data = labels_data.loc[unique_hadmids]
  labels_data['mortality'] = 1 - labels_data['dateofdeath'].isna().astype(int)
  cols_to_drop = ['dateofdeath'] ; labels_data.drop(columns=cols_to_drop,inplace=True)

  return labels_data.reset_index().to_csv(path + 'preprocessed_labels.csv', index=False)
