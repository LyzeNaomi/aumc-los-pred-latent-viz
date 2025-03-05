#import dependencies
import pandas as pd
import numpy as np
from itertools import islice
import os

data_path = ''
AMSUMC_timeseries = 'numericitems.zip'
AMSUMC_admissions = 'admissions.csv'
cols = ['admissionid', 'itemid', 'item', 'tag', 'value', 'unit', 'measuredat', 'registeredat', 'updatedat','islabresult', 'fluidout']


def configure_preprocess(AMSUMC_path, cols, perc_labs=25, perc_non_labs=35, verbose=False, test=False, save_intermediate = False): 
    ''' 
    Main function that reads in the data and outlines the dataflow and subsequent function calls
    '''
    if verbose:
      print('Starting chunkwize slicing')

    sliced_data = slice_chunkwize(AMSUMC_path=AMSUMC_path, cols=cols, verbose=verbose, threshold_min_stay=5, test=test, save_intermediate=save_intermediate)

    non_lab_data, lab_data = split_and_reduce(sliced_data, perc_labs, perc_non_labs, verbose=verbose) #this returns the reduced lab and non-lab data
    del(sliced_data)
    
    if save_intermediate: 
        lab_data.to_csv(AMSUMC_path + 'reduced_lab_data.csv.gz', header = True, index=False, compression='gzip')
        non_lab_data.to_csv(AMSUMC_path + 'reduced_full_non_lab_data.csv.gz', header = True, index=False, compression='gzip')
    
    timeseries_labs, timeseries_non_labs = gen_timeseriesfile(lab_data, non_lab_data, verbose=verbose)
    del(lab_data, non_lab_data)

    resampled_timeseries = resample_mask(AMSUMC_path=AMSUMC_path, timeseries_labs=timeseries_labs, timeseries_non_labs=timeseries_non_labs, verbose=verbose, test=test)
    del(timeseries_labs, timeseries_non_labs)

    preprocessed_timeseries = scale_and_fill(AMSUMC_path, resampled_timeseries, verbose=verbose)
    del(resampled_timeseries)

    if verbose:
      print('Saving preprocessed timeseries data')
    
    preprocessed_timeseries.to_csv(AMSUMC_path + 'preprocessed_timeseries.csv.gz', compression='gzip')

    return

def slice_chunkwize(AMSUMC_path, cols, verbose, test, save_intermediate, threshold_min_stay=5):
    '''
    Slice the raw data such that only patients with a LoS > predefined threshold (default = 5) are preserved.
    In addition, slice the data such that for each admission data is available for -24h (prior to admission) - LoS
    '''
    
    if verbose:
      print('Creating chunks')
    
    #define chunk
    chunksize = int(9.776256e+08 / 200)
    chunked_data = pd.read_csv(AMSUMC_path + AMSUMC_timeseries, usecols=cols, compression='zip', encoding = 'unicode_escape', dtype={'registeredat': 'float64',
      'value': 'float64'}, iterator = True, chunksize = chunksize)

    #slice chunk
    if verbose:
      print('Slicing data')
    
    admission = pd.read_csv(AMSUMC_path + AMSUMC_admissions, usecols=['admissionid', 'lengthofstay'])
    admission = admission.loc[admission.lengthofstay > threshold_min_stay]

    chunk_n = 0
    header = True
    for chunk in chunked_data:
      data = chunk.loc[chunk.admissionid.isin(admission.admissionid.to_list())]
      data = data.loc[data.measuredat >= -8.64e+7]
    
      for id in data.admissionid.unique():
        los = admission.loc[admission.admissionid == id, 'lengthofstay'].values[0]
        data_id = data[data.admissionid == id].loc[data[data.admissionid == id].measuredat <= los*3.6e+6]
        data_id.to_csv(AMSUMC_path + 'sliced_data.csv.gz', mode = 'a', header = header, index=False, compression='gzip') ; header = False     
    
      if verbose:
        print(f'Finished slicing chunk: {chunk_n}') 
      
      chunk_n = chunk_n + 1
      
      if test & (chunk_n > 0): 
          print('Test threshold has been reached')
          sliced_data = pd.read_csv(AMSUMC_path + 'sliced_data.csv.gz', compression='gzip')
          if not save_intermediate: 
            os.remove(AMSUMC_path + 'sliced_data.csv.gz') 
          return sliced_data

    if verbose:
      print('Removed all patients that do not meet LoS threshold') 
      print('Sliced data in the range -24 - LoS')
      print('Reading sliced data from the compressed file')
    
    sliced_data = pd.read_csv(AMSUMC_path + 'sliced_data.csv.gz', compression='gzip')

    if not save_intermediate: 
       os.remove(AMSUMC_path + 'sliced_data.csv.gz') 

    return sliced_data    

    
def split_and_reduce(full_data, perc_labs, perc_non_labs, verbose):
    '''
    First, split the data into labs and non labs data to obtain more manageable chunks for processing
    Subsequently reduce the data such that the items (variables) included in the subset are observed for at least
    a certain (user defined) amount of the admissions (procentually)
    '''
    if verbose:
      print('Splitting data into labs and non labs')
    
    #split data in labs and non labs
    non_lab_data = full_data.loc[full_data.islabresult == 0]
    lab_data = full_data.loc[full_data.islabresult == 1]
    
    #delete the full data to reduce memory pressure
    del(full_data)
   
    #reduce the data and pass to main function
    if verbose:
      print('Reducing non lab data')

    reduced_non_lab_data = reduce_data(non_lab_data, percentage = perc_non_labs)

    if verbose:
      print('Reducing lab data')

    reduced_lab_data = reduce_data(lab_data, percentage = perc_labs)
    return reduced_non_lab_data, reduced_lab_data


def reduce_data(data_to_reduce, percentage = 35):
    '''
    In order to reduce the data (labs or non labs) based on a set (user defined) 
    threshold for the observed frequency of the itemids we first caolculate those frequencies on the admission+itemid level. 
    Subsequently, the itemids that meet the threshold are selected and the other items are dropped from the data
    '''
    count = get_count(data_to_reduce)
            
    selected_itemids = itemids_selection(u_item_admid = count, perc_record_per_item = 'n_perc_item_recorded', item_record_threshold = percentage)
    print(len(selected_itemids))
    reduced_data = data_to_reduce.loc[data_to_reduce.itemid.isin(selected_itemids)]
    print(reduced_data.shape)
    print(len(reduced_data.itemid.unique()))
    return reduced_data


def get_count(data):
    '''
    Here, we calculate for each item, the frequency of observance with respect to 
    the total admissions contained in the data (labs or non labs)
    As such, for each itemid we obtain a percentage that represents: 
    "the percentage of admissions for which a given item(id) has been observed"
    which is then used to match against the threshold in the parent function for the itemid selection
    '''
    l_adms = len(data.admissionid.unique())
    data = data.groupby(['admissionid','itemid'])['itemid'].count().reset_index(name ='itemcount')
    data = data.groupby(['admissionid', 'itemid'])['itemcount'].sum().reset_index(name ='itemcount')
    data = data.groupby(['itemid'])['itemid'].count().reset_index(name='total_unique_itemcount')
    data['n_perc_item_recorded'] = 100*(data.total_unique_itemcount/l_adms)
    return data


def itemids_selection(u_item_admid, perc_record_per_item, item_record_threshold):
    '''
    Select all itemids that are observed at least in a certain percentage of the 
    admissions (threshold is user defined but defaults to 35 in the parent function)
    '''
    n_perc_item_recorded = u_item_admid[u_item_admid[perc_record_per_item] >= item_record_threshold]['itemid'].unique()
    return set(n_perc_item_recorded)


def gen_timeseriesfile(lab_data, non_lab_data, verbose):
    '''
    For both the lab data and the non lab data, generate the timeseries file in wide format. 
    The lab data is sufficiently small to do so efficiently at once. The non-lab data is
    first split into 10 unique parts based on the unique itemids and is then converted chunkwize
    into the wide format. Ultimately returning the full wide data for labs and non labs
    '''
    #Generate the wide format at once for labs and blockwize for non-Labs
    lab_data.loc[lab_data.item == 'APTT  (bloed)', 'item'] = 'APTT (bloed)'
    timeseries_labs = reconfigure_timeseries(timeseries = lab_data, offset_column='measuredat', value_column ='value', hadmid = 'admissionid',
                                     time_unit = 'ms', feature_column = 'item')
    del(lab_data)

    if verbose:
      print('Reconfigured lab data') 

    #do the blockwise timeserie wide generation here for the non labs
    unique_items_nl = non_lab_data.itemid.unique()
    split_items = np.array_split(unique_items_nl, 10)

    timeseries_non_labs = pd.DataFrame() 
    for i in range(len(split_items)):
        subset = non_lab_data.loc[non_lab_data.itemid.isin(split_items[i])]
        subset_wide = reconfigure_timeseries(timeseries = subset, offset_column='measuredat', value_column ='value', hadmid = 'admissionid',
                                     time_unit = 'ms', feature_column = 'item')
        non_lab_data = non_lab_data.loc[~non_lab_data.itemid.isin(split_items[i])] #needed to gradually reduce df size and stress on memory
        timeseries_non_labs = timeseries_non_labs.append(subset_wide)

        if verbose:
          print(f'Reconfigured non lab data chunk: {i}')

    return timeseries_labs, timeseries_non_labs


def reconfigure_timeseries(timeseries, offset_column, value_column, hadmid, time_unit = 'ms', feature_column = None, test=None, nrows=None):
    '''
    Reconfiguration of the long format timeserie data to the wide format
    Essentially taking a pivot of the long formatted data
    '''
    if test: 
      timeseries = timeseries.iloc[:nrows]

    timeseries.set_index([hadmid, pd.to_timedelta(timeseries[offset_column], unit=time_unit)], inplace=True)
    timeseries.drop(columns=offset_column, inplace=True)
    if feature_column is not None:
        timeseries = timeseries.pivot_table(columns=feature_column, index=timeseries.index, 
                                            values = value_column)
    timeseries.index = pd.MultiIndex.from_tuples(timeseries.index, names=['hadmid', 'time'])
    return timeseries


def resample_mask(AMSUMC_path, timeseries_labs, timeseries_non_labs, test=False, verbose=True):
    '''
    Parent function for the resampling and masking to orchestrate resampling and
    masking in a chunkwize manner in order to reduce stress on memory
    Code based on Rocheteau
    '''
    preprocessed_ts = pd.DataFrame()
    hadmid = timeseries_labs.index.unique(level=0) 
    size = 5000
    gen_chunks = gen_patient_chunk(hadmid, size=size)
    i = size
    header = True
    print('==> Starting main processing loop...')

    for patient_chunk in gen_chunks:
        merged = timeseries_labs.loc[timeseries_labs.index.get_level_values(0).isin(patient_chunk)].append(timeseries_non_labs.loc[timeseries_non_labs.index.get_level_values(0).isin(patient_chunk)], sort=False)

        processed_ts_chunk = resample_and_mask(timeseries = merged, AMSUMC_path=AMSUMC_path, header=header, fill_rem='nothing', 
                                                    mask_decay=True, decay_calc_method='division', decay_rate=4/3, test=test,
                        verbose=verbose, length_limit=24*14)
        
        #preprocessed_ts = preprocessed_ts.append(processed_ts_chunk)
        processed_ts_chunk.to_csv(AMSUMC_path + 'preprocessed_timeseries.csv', mode='a', header=header)
        
        print('==> processed ' + str(i) + ' patients...')
        header = False; i+=size   
    return #preprocessed_ts


def gen_patient_chunk(patients, size):
    '''
    Generate an iterable with patient chunks to do the resampling over
    '''
    it = iter(patients)
    chunk = list(islice(it, size))
    while chunk:
        yield chunk
        chunk = list(islice(it, size))


def scale_and_fill(AMSUMC_path, timeseries, verbose):
    '''
    Perform scaling over all the data and clip to remove outliers
    Code in part based on Rocheteau
    '''
    
    if verbose:
      print('Scaling and clipping data')
    
    
    all_cols = [col for col in timeseries.columns]
    mask_cols = [col for col in timeseries.columns if '_mask' in col]
    no_scale_cols = mask_cols + ['hadmid', 'time']
    to_scale_cols = list(set(all_cols).difference(set(no_scale_cols)))

    #Based on Rocheteau
    quantiles = timeseries[to_scale_cols].quantile([0.05, 0.95])
    maxs = quantiles.loc[0.95]
    mins = quantiles.loc[0.05]
    timeseries[to_scale_cols] = (2 * (timeseries[to_scale_cols] - mins) / (maxs - mins) - 1)
    
    # we then need to make sure that ridiculous outliers are clipped to something sensible
    timeseries[to_scale_cols] = timeseries[to_scale_cols].clip(lower=-4, upper=4)  # room for +- 3 on each side, as variables are scaled roughly between 0 and 1
    timeseries[to_scale_cols] = timeseries[to_scale_cols].fillna(0)
    timeseries.to_csv(AMSUMC_path + 'scaled_preprocessed_timeseries.csv', index=False)
    return #timeseries


def resample_and_mask(timeseries, AMSUMC_path, header, fill_rem, mask_decay:bool, decay_calc_method, decay_rate=4/3, test=False,
                      verbose=True, length_limit=24*14): 
  '''
  Resample the data to an hourly format, calculate the decaying mask for non observed entries and 
  perform forward filling on a per admission id basis
  Part primarily based on Rocheteau
  '''
  if test:
    mask_decay = True
    verbose = True
  if verbose:
    print('Resampling to 1 hour intervals...')
  #take the mean of any duplicate index entries for unstacking
  timeseries = timeseries.groupby(level=[0,1]).mean()

  timeseries.reset_index(level=1, inplace=True)
  timeseries.time = timeseries.time.dt.ceil(freq='H')
  timeseries.set_index('time', append=True, inplace=True)
  timeseries.reset_index(level=0, inplace=True)
  resampled = timeseries.groupby('hadmid').resample('H', closed='right', label='right').mean().drop(columns='hadmid')
  del(timeseries)

  def apply_mask_decay(mask_bool):
    mask = mask_bool.astype(int) #non na's 
    mask.replace({0: np.nan}, inplace=True)
    inv_mask_bool = ~mask_bool #count of na's
    count_non_measurements = inv_mask_bool.cumsum()-inv_mask_bool.cumsum().where(mask_bool).ffill().fillna(0) #
    if decay_calc_method == 'division':
      decay_mask = mask.ffill().fillna(0) / (count_non_measurements * decay_rate).replace(0, 1)
    elif decay_calc_method == 'exponential_decay':
      decay_mask = mask.ffill().fillna(0) / (decay_rate**count_non_measurements).replace(0, 1)
    return decay_mask

  if mask_decay:
    if verbose:
      print('Calculating mask decay features...')
    mask_bool = resampled.notnull() #count filled entries.
    mask = mask_bool.groupby('hadmid').transform(apply_mask_decay)
    del (mask_bool)
  else:
    if verbose:
      print('Calculating binary mask features...')
    mask = resampled.notnull()
    mask = mask.astype(int)

  if verbose:
    print('Filling missing data forward...') 
  resampled = resampled.groupby(['hadmid']).fillna(method='ffill')
  # simplify the indexes of both tables
  mask = mask.rename(index=dict(zip(mask.index.levels[1],
                                    mask.index.levels[1].days*24 + mask.index.levels[1].seconds//3600)))
  mask.to_csv(AMSUMC_path+'mask.csv')
  resampled = resampled.rename(index = dict(zip(resampled.index.levels[1],
                                            resampled.index.levels[1].days*24 + resampled.index.levels[1].seconds // 3600)))
  #clip to length_limit
  if length_limit is not None:
    within_length_limit = resampled.index.get_level_values(1) < length_limit
    resampled = resampled.loc[within_length_limit]
    mask = mask.loc[within_length_limit]

  if verbose and fill_rem == 'nothing':
    print('No filling of further missing information')
  
  if verbose and fill_rem == 'bfill':
    print('Backward filling in remaining values')
    resampled = resampled.groupby(level=0).fillna(method = 'bfill') 

  # rename the columns in pandas for the mask so it doesn't complain
  mask.columns = [str(col) + '_mask' for col in mask.columns]

  # merge the mask with the features
  final = pd.concat([resampled, mask], axis=1)
  final.reset_index(level=1, inplace=True)
  final = final.loc[final.time > 0]

  if verbose:
      print('Current chunk is processed and being appended to the main dataframe')
  return final

configure_preprocess(AMSUMC_path=data_path, perc_labs=25, perc_non_labs=35, cols=cols,
                      verbose=True, test=True, save_intermediate=False)