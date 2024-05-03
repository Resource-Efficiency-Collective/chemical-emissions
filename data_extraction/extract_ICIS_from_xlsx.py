"""This script converts a series of Excel files downloaded from ICIS with all Countries selected into
 summary files for 'Capacity', 'Statistic Production', 'Import', 'Export', 'Consumption'"""

import os
import numpy as np
import pandas as pd

filepath = "D:/data/ICIS_data/world_chemical/2050_update/"
outpath = "D:/data/ICIS_data/world_chemical/2050_update/"
categories = ['Capacity', 'Statistic Production', 'Import', 'Export', 'Consumption']

Capacity, Statistic_Production, Import, Export, Consumption = [pd.DataFrame() for i in range(5)]
for file in os.listdir(filepath):
    in_file = pd.ExcelFile(filepath+file)
    data = pd.read_excel(in_file, sheet_name=in_file.sheet_names[2], skiprows=2)[1:].reset_index(drop=True)
    for data_type in categories:
        if data_type in list(data.PRODUCT):
            st_index = data[data['PRODUCT'] == data_type].index[0]+1
            beyond = data[st_index:]
            nan_locs = (np.where(['nan' in str(i) for i in beyond['PRODUCT']])+st_index)[0]
            for index in nan_locs:
                if index+1 > len(data['PRODUCT'])-1 or data['PRODUCT'][index+1] in categories:
                    end_index = index-1
                    break
            locals()[data_type.replace(' ', '_')] = pd.concat((locals()[data_type.replace(' ', '_')], data[st_index:end_index].dropna(subset=['COUNTRY/TERRITORY'])), axis=0)
        else:
            continue

Statistic_Production.to_csv(outpath+'production.csv', index=False)
Import.to_csv(outpath+'imports.csv', index=False)
Export.to_csv(outpath+'exports.csv', index=False)
Capacity.to_csv(outpath+'capacity.csv', index=False)
Consumption.to_csv(outpath+'consumption.csv', index=False)