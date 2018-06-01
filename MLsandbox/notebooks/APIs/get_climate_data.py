
# coding: utf-8

import pandas as pd
import json
import requests
import datetime
from collections import defaultdict
import sys
import logging
from time import sleep, strftime
import numpy as np

global metadata


def load_metadata(folder):
    
    metadata = {}
    
    if folder[-1] == '/':
        folder = folder[:-1]
        
    # get information about available datatypes and their ids
    with open(folder + '/data_types_clean.json') as fp:
        metadata['data_json'] = json.load(fp)

    # create a dictionary
    metadata['data_list'] = {k:v for k,v in metadata['data_json'].items()}

    # for clarity and searchability, make an inverted dictionary indexed on id:
    metadata['data_list_inv'] = {item[1]['id']:item[0] 
                                 for item in metadata['data_list'].items()}
    
    # load available datasets and their associated data types
    with open('../../climate_data/datasets_by_datatypes.json') as fp:
        metadata['datasets_dict'] = json.load(fp)
    
    # load datatypes by dataset (for setting column names in df)
    with open('../../climate_data/datatypes_by_datasets_all.json') as fp:
        metadata['dt_by_ds'] = json.load(fp)

    # get list of locations
    with open('../../climate_data/data_locations.json') as fp:
        metadata['loc_list'] = json.load(fp)
    
    return metadata

def make_request(url, headers, params):
    logger = logging.getLogger('api_logger')
    try:
        sleep(np.random.randint(0,10))
        r = requests.get(url, headers = headers, params = params)
        r.raise_for_status()
        print r
        return r
    except requests.exceptions.HTTPError as e:
        logger.info(e)
        logger.info(r.raise_for_status)
        if e.response.status_code ==429: #remove token for this session
            logger.info(('popped token {0}.  There are {1} token(s) left'
                       ).format(tokens[tok_number]['name'], len(tokens)-1))
            tokens.pop(tok_number)
            return #make_request(url, headers, params) 
    return r

def add_to_df(data, df, output_file):
    logger = logging.getLogger('api_logger')

    # group observations, take the mean over all years for each station/datatype
    # and unstack into a regular dataframe
    temp = pd.read_json(json.dumps(data['results']), orient = 'records')
    temp['datatype_id'] = temp['datatype']
    temp['datatype'] = temp['datatype_id'].map(metadata['data_list_inv'])
    temp_grouped = (pd.DataFrame(temp.groupby(by=['station', 'datatype_id']).mean())
                    .reset_index()
                    .pivot(index='station', columns='datatype_id', values='value')
                    .reset_index())
    
    # append to existing df
    try:
        df = pd.concat([df,temp_grouped])
    except:
        print df.head()
        print "could not concat to this df"
    # printout for monitoring
    logger.debug("last record: ")
    logger.debug(data['results'][-1])
    logger.info("number of records found: %s" %str(len(data['results'])))
    logger.info("df has column length: %s" %str(len(df.columns)))
    logger.info("and number of records: %s" %str(len(df)))
    
    #print json.loads(df.reset_index().to_json(orient='records'))

    df.iloc[-(len(temp_grouped)):, :].to_csv(output_file[:-5]+'.csv', index=False, header=None, mode='a')
 
    if len(df) > 1000:
        with open(output_file, 'w') as f:
            json.dump(json.loads(df.reset_index().to_json(orient='records')), f) 
        c = int(output_file[:-5].split('_')[-1])
        print "dumped to %s" % output_file
	logger.info("dumped to %s" % output_file)
	
        df = pd.DataFrame()
        output_file = '_'.join(output_file.split('_')[:-1])+ '_' + str(int(output_file[:-5].split('_')[-1])+1) + '.json'

    return df, output_file


metadata = load_metadata('../../climate_data/')
'''df = pd.DataFrame()
ds_of_interest = ['GSOY']
step_size = 100
startdate = '2003-02-03'
enddate = '2011-04-05'
'''

def main(folder, ds_of_interest, step_size, startdate, enddate, startloc, endloc):

    output_file = str(folder + 'climate_' +  str(ds_of_interest[0])
                      + '_' + str(step_size) + '_' + str(startdate.strip('-')) + '_' + str(enddate.strip('-')) 
                      + '_' + str(startloc) + '_' + str(endloc) + '_0.json')

    file_num = 0
    print output_file

    # start logging
    output_filename = output_file[:-5] + '.log'  # log file for debugging
    logging.basicConfig(level=logging.INFO, filemode = 'w', filename = output_filename,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('api_logger')
    logger.info('starting logging')
    handler = logging.handlers.RotatingFileHandler(output_filename, maxBytes=10000, backupCount=10)
    
    # open folder of tokens for rotation and set number of requests for session to 0
    global tokens
    with open(folder + 'tokens.txt') as fp:
        tokens = json.load(fp)
    for item in tokens:
        item['requests']=0

    # API requires one dataset at a time to be queried
    for dataset in ds_of_interest:
        
        logger.info(('Configuration: ds_of_interest = {0}\n step_size = {1}\n startdate = {2}\n enddate = {3}\n startloc = {4}\n endloc = {5}\n'
                   ).format(dataset, str(step_size), startdate, enddate, str(startloc), str(endloc)))
        
        # make a new dataframe with the appropriate columns for this datatype
        df = pd.DataFrame(columns = [datatype['id'] for datatype in metadata['dt_by_ds'][dataset]])
        df.to_csv(output_file[:-5]+'.csv', index=False, header=1, mode='a')

        # In order to stay under the url character limit, set a step size of <50 locations per request
        for loc_offset in range(startloc, 
                                (endloc + step_size),
                                step_size):

            i = 0  # number of offset iterations

            logger.info('\n==============\n ZIPs in loc_list starting at index: ' + str(loc_offset))
            
            # rotate tokens and set header/parameters for request
            global tok_number
            tok_number = (loc_offset/step_size)%(len(tokens))
            headers = {'token' : tokens[tok_number]['token']}
            logger.info(("using {0}'s token; request #{1}")
                        .format(tokens[tok_number]['name'], 
                                tokens[tok_number]['requests']))

            dtparams = ['AWND', 'DP10', 'DSNW', 'DT32', 'DX90', 'EVAP']
            params = {'datasetid':'GSOY', 
                      'locationid':[item['id'] for item in
                                    metadata['loc_list'][loc_offset:loc_offset+50]],
                      'limit':1000, 'offset': i*1000,
                      'startdate':startdate, 'enddate':enddate}

            #submit request and increment number of requests for that token
            r = make_request('https://www.ncdc.noaa.gov/cdo-web/api/v2/data', headers, params)
            tokens[tok_number]['requests'] += 1
           
            try:
                data = json.loads(r.text)

                #check if number of records retrieved hit the limit
                num_records_found = len(data['results'])

                df, output_file = add_to_df(data, df, output_file)

                while num_records_found >= 1000: # if we hit our limit of records, go back for more
                    i += 1
                    params['offset']+=1000
                    
                    logger.info("Offset set to %s" %params['offset'])
                    logger.debug("------")
                    logger.debug(params)
                    logger.debug("------")
                    r = make_request('https://www.ncdc.noaa.gov/cdo-web/api/v2/data', headers, params)
                    tokens[tok_number]['requests'] += 1
                    try:
                        data = json.loads(r.text)
                        num_records_found = len(data['results'])
                        df, output_file = add_to_df(data, df, output_file)
                    except KeyError:
                        logger.info("KEYERROR: ")
			logger.info(r.text)
            except:
                print r
	        if KeyError:
                    logger.info("KEYERROR: ")
                
        with open(output_file, 'w') as f:
            json.dump(json.loads(df.reset_index().to_json(orient='records')), f)
        c = int(output_file[:-5].split('_')[-1])
        print "dumped to %s" % output_file
        logger.info("dumped to %s" % output_file)

        df = pd.DataFrame()
        output_file = '_'.join(output_file.split('_')[:-1])+ '_' + str(int(output_file[:-4].split('_')[-1])+1) + '.csv'    


    return 'success'



if __name__ == '__main__':
    folder, ds_of_interest, step_size, startdate, enddate, startloc, endloc = sys.argv[1:8]
    main(folder, eval(ds_of_interest), int(step_size), startdate, enddate, int(startloc), int(endloc))

