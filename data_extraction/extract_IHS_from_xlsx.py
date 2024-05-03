"""Script to append IHS data to feather files"""
import sys
import pandas as pd
import numpy as np


def del_empty_cols(df):
    return df.dropna(axis=1, how='all').reset_index(drop=True)


class ProcessData(object):
    def __init__(self, in_file):
        self.data = del_empty_cols(pd.read_excel(in_file, header=6, skipfooter=1)).dropna(axis=0, how='all')
        self.products = del_empty_cols(self.data.loc[self.data['Type'] == 'Product'])
        self.materials = del_empty_cols(self.data.loc[self.data['Type'] != 'Product'])

    def format_products(self):
        prod_rename = {'Variable Cost': 'Variable Cost (¢/kg)', 'Fixed Costs': 'Fixed Costs (¢/kg)',
                       'Overhead + Tax, Ins.': 'Overhead + Tax, Ins. (¢/kg)', 'Plant Cash Cost ': 'Plant Cash Cost (¢/kg)',
                       'Depreciation': 'Depreciation (¢/kg)', 'Plant Gate Costs ': 'Plant Gate Costs (¢/kg)',
                       'G&A, Sales, Res.': 'G&A, Sales, Res. (¢/kg)', 'Production Cost ': 'Production Cost (¢/kg)',
                       'ROI (15%)': 'ROI (15%) (¢/kg)', 'Product Value ': 'Product Value (¢/kg)',
                       'Product Price ': 'Product Price (¢/kg)'}
        return self.products.rename(columns=prod_rename)

    def format_materials(self):
        merged = pd.merge(self.materials, self.products, how='left', on=['Code'])
        merged['Unit_conv'] = ['MM '] * len(merged) + merged['Unit'] + ['/yr'] * len(merged)
        merged['Uncertainty'] = [np.nan] * len(merged)
        merged['Provenance'] = ['IHS PEP'] * len(merged)
        short_list = ['Code', 'Data Version_x', 'Name_x', 'Process_x', 'Research Year_x', 'Geography_x',
                      'Base Capacity (MM unit/year)', 'Unit_conv', 'Name_y', 'Unit Consumption', 'Consumption Unit',
                      'Uncertainty', 'Provenance']
        long_list = short_list + [i for i in merged.columns if
                                  i not in short_list + ['Geography_y', 'Data Version_y', 'Research Year_y',
                                                         'Process_y',
                                                         'Unit']]
        rename_dict = {'Data Version_x': 'Data Version', 'Name_x': 'Source', 'Process_x': 'Target',
                       'Research Year_x': 'Research Year',
                       'Geography_x': 'Geography', 'Base Capacity (MM unit/year)': 'Plant capacity',
                       'Unit_conv': 'Capacity unit', 'Name_y': 'Product',
                       'Unit Consumption': 'Value', 'Consumption Unit': 'Value unit', 'Type_x': 'Source type',
                       'Variable Cost_x': 'Source cost (¢/kg)',
                       'Investment (MM US$)': 'Product Investment (MM US$)', 'Type_y': 'Product Type',
                       'Variable Cost_y': 'Product variable cost (¢/kg)', 'Fixed Costs': 'Product fixed costs (¢/kg)',
                       'Overhead + Tax, Ins.': 'Product Overhead + Tax, Ins. (¢/kg)',
                       'Plant Cash Cost ': 'Product Plant Cash Cost (¢/kg)', 'Depreciation': 'Product Depreciation (¢/kg)',
                       'Plant Gate Costs ': 'Product Plant Gate Costs (¢/kg)', 'G&A, Sales, Res.': 'Product G&A, Sales, Res. (¢/kg)',
                       'Production Cost ': 'Production Cost (¢/kg)', 'ROI (15%)': 'Product ROI (15%) (¢/kg)',
                       'Product Value ': 'Product Value (¢/kg)', 'Product Price ': 'Product Price (¢/kg)'}

        return merged[long_list].rename(columns=rename_dict)


def main(in_file, append, material_file, product_file):
    data = ProcessData(in_file)
    if append:
        pd.read_feather(material_file).append(data.format_materials()).drop_duplicates(ignore_index=True)\
            .reset_index(drop=True).to_feather(material_file)
        pd.read_feather(product_file).append(data.format_products()).drop_duplicates(ignore_index=True)\
            .reset_index(drop=True).to_feather(product_file)
        print('Append successful')
    else:
        data.format_materials().to_feather(material_file)
        data.format_products().to_feather(product_file)
        print('File creation successful')


if __name__ == "__main__":
    append = True
    material_file, product_file = "C:/IHS_data/materials.feather", "C:/IHS_data/products.feather"
    in_file = sys.argv[1]
    main(in_file, append, material_file, product_file)


