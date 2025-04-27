## Functions used in process_iterations.ipynb file.

import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

input_path = '../data/'

def update_ef_dfs(ei_emissions, match_list_ei, cm_emissions, match_list_cm, iteration_ef_updates,
                  current_chemical_names):
    """Update EFs with new values from iteration to use in next iteration"""
    iteration_ef_updates = iteration_ef_updates[iteration_ef_updates['CO2e_100a'] != 0]  # Remove non produced items
    iteration_ef_updates['PRODUCT'] = iteration_ef_updates['PRODUCT'].str.replace('BTX', 'REFORMATE')

    iteration_ef_updates['Source'] = iteration_ef_updates['PRODUCT'].str.lower()
    iteration_ef_updates['generalComment'] = 'C-THRU calculation'
    iteration_ef_updates['location'] = 'GLO'
    iteration_ef_updates['Product'] = iteration_ef_updates['PRODUCT'].str.lower()
    iteration_ef_updates = iteration_ef_updates.drop(columns=['PRODUCT'])
    iteration_ef_updates = iteration_ef_updates[list(ei_emissions.columns)]

    ei_updated = pd.concat((iteration_ef_updates, ei_emissions)).drop_duplicates(subset=['Source'], keep='first')

    update_matches = iteration_ef_updates.copy()
    update_matches['ei'], update_matches['IHS'] = update_matches['Source'], update_matches['Source']
    ei_update_matches = match_list_ei[[i in update_matches['IHS'].unique() for i in match_list_ei['IHS']]]
    ei_update_matches = ei_update_matches[ei_update_matches['ei'] != '0']
    ei_update_matches = ei_update_matches.set_index('ei')['IHS'].to_dict()
    match_list_ei['ei'] = match_list_ei['ei'].replace(ei_update_matches)
    # ei_equivalents = match_list_ei[match_list_ei['IHS'].isin(current_chemical_names.str.lower())]
    # ei_current_emissions = ei_equivalents.merge(ei_emissions, left_on='ei', right_on='Source', how='inner')
    # merged_updates = ei_current_emissions.merge(iteration_ef_updates, left_on='IHS', right_on=iteration_ef_updates['PRODUCT'].str.lower(), how='inner')
    # ei_updates = merged_updates[['Source', 'generalComment', 'location']+[col for col in merged_updates.columns if '_y' in col]]
    # ei_updates['location'], ei_updates['generalComment'] = 'GLO', 'C-THRU calculation'
    # ei_updates.columns = [col.replace('_y', '') for col in ei_updates.columns]
    #
    # ei_emissions = pd.concat((ei_updates, ei_emissions[~ei_emissions['Source'].isin(ei_updates['Source'])]))

    cm_emissions = cm_emissions[~cm_emissions['Source'].isin(
        match_list_cm[match_list_cm['IHS'].isin(current_chemical_names.str.lower())]['cm'])]

    # match_list_cm.loc[match_list_cm[match_list_cm['IHS'].isin(current_chemical_names.str.lower())].index, 'cm'] = '0'

    current_chem_matches = match_list_ei.loc[
        list((match_list_ei[match_list_ei['IHS'].isin(current_chemical_names.str.lower())]['ei'] != '0').index)]['IHS']

    # match_list_cm.loc[match_list_cm[match_list_cm['IHS'].isin(current_chem_matches.values)].index, 'cm'] = '0'
    match_list_cm.loc[match_list_cm['cm'].isin(
        match_list_cm[match_list_cm['IHS'].isin(current_chem_matches.str.lower())]['cm']), 'cm'] = '0'

    return ei_updated, match_list_ei, cm_emissions, match_list_cm


def get_updated_efs(facility_production, current_ifa, emissions_weighted, current_chemical_names, cf_subset):
    production = pd.concat((facility_production, current_ifa.drop(columns=['Conv_name'])))
    current_group_prod = production[['PRODUCT', '2020', '2020_sigma']][
        production['PRODUCT'].isin(current_chemical_names)].groupby('PRODUCT').sum()

    current_group_emissions = emissions_weighted[['PRODUCT', 'Gas', '2020', '2020_sigma']][
        emissions_weighted['PRODUCT'].isin(current_chemical_names)].groupby(['PRODUCT', 'Gas']).sum()

    merged = current_group_emissions.reset_index().merge(current_group_prod, on='PRODUCT', how='left')

    merged['EF'] = merged['2020_x'].values / merged['2020_y'].values
    merged['EF_sigma'] = merged['2020_sigma_x'].values / merged['2020_y'].values

    # Pivot pandas df to have product as rows, gas as columns and EF as values
    efs = merged.pivot_table(index='PRODUCT', columns='Gas', values='EF').reset_index()
    ef_sigmas = merged.pivot_table(index='PRODUCT', columns='Gas', values='EF_sigma').reset_index()
    new_efs = efs.merge(ef_sigmas, on='PRODUCT', suffixes=('', '_sigma'))

    non_produced_ef = cf_subset[~cf_subset['Product'].isin(new_efs['PRODUCT'])]
    non_produced_ef = non_produced_ef[['Product'] + [col for col in non_produced_ef.columns if 'ihs' in col]].rename(
        columns={'Product': 'PRODUCT'}).drop(columns=['ihs_match'])
    non_produced_ef.columns = pd.Series(non_produced_ef.columns).str.replace('ihs_cradle-to-out-gate ', '').str.replace(
        ',  allocation factor', '').str.replace(',  allocation ', '_')
    non_produced_ef[non_produced_ef.columns[1:]] = non_produced_ef[non_produced_ef.columns[1:]].astype(float)
    non_produced_ef = non_produced_ef.groupby('PRODUCT').mean().reset_index()

    return pd.concat((new_efs, non_produced_ef))


def merge_weighted_ethylene(facility_emissions, ethylene_conv, dbs=None, names=None, emission_val_cols=None):
    if dbs is None:
        dbs = ['combined_', 'ihs_cradle-to-out-gate ', 'Feedstock ', 'Primary chemicals ', 'Intermediates ',
               'Indirect Utilities ', 'Direct Utilities ', 'Direct Process ', 'Electricity ', 'Thermoplastics ',
               'N-fertilisers ', 'Solvents, additives & explosives ', 'Thermosets, fibre & elastomers ',
               'Other downstream ']
    if names is None:
        names = ['EI & CM', 'IHS CtOG', 'Feedstock', 'Primary chemicals', 'Intermediates', 'Direct Utilities',
                 'Indirect Utilities', 'Direct Process', 'Electricity', 'Thermoplastics', 'N-fertilisers',
                 'Solvents, additives & explosives', 'Thermosets, fibre & elastomers', 'Other downstream']
    if emission_val_cols is None:
        emission_val_cols = ['CO2e_20a', 'CO2e_100a', 'CO2e_500a', 'Carbon dioxide', 'Carbon monoxide', 'Chloroform',
                             'Dinitrogen monoxide', 'Ethane', 'Methane', 'Nitric oxide', 'Nitrogen fluoride',
                             'Perfluoropentane', 'Sulfur hexafluoride']

    base_cols = list(ethylene_conv.columns[:7])

    ethylene_vals = pd.DataFrame()
    ethylene_sigmas = pd.DataFrame()
    ethylene_conv['conversion'] = [i.replace(',  allocation factor', '').replace(',  allocation ', '_') for i in
                                   ethylene_conv['conversion']]

    for db, name in zip(dbs, names):
        for gas in emission_val_cols:
            df = ethylene_conv[ethylene_conv['conversion'] == db + gas]
            df['Gas'] = gas
            df['Type'] = name
            ethylene_vals = pd.concat((ethylene_vals, df), axis=0)

            df_sigma = ethylene_conv[ethylene_conv['conversion'] == db + gas + '_sigma']
            df_sigma['Gas'] = gas
            df_sigma['Type'] = name
            ethylene_sigmas = pd.concat((ethylene_sigmas, df_sigma), axis=0)

    ethylene_weighted = ethylene_vals.merge(ethylene_sigmas, on=base_cols + ['Gas', 'Type'], how='left',
                                            suffixes=('', '_sigma')).reset_index()
    #
    # #ethylene_weighted.columns.name = None
    ethylene_weighted = ethylene_weighted.fillna(0).drop(columns=['conversion', 'conversion_sigma', 'index'])

    ethylene_weighted[['COUNTRY/TERRITORY', 'STATE', 'COMPANY', 'SITE', '#', 'START_YR', 'Type', 'Gas']] = \
    ethylene_weighted[['COUNTRY/TERRITORY', 'STATE', 'COMPANY', 'SITE', '#', 'START_YR', 'Type', 'Gas']].astype(str)

    eth_ems = facility_emissions[facility_emissions['PRODUCT'] == 'ETHYLENE']

    emissions_merged = eth_ems.merge(ethylene_weighted,
                                     on=['COUNTRY/TERRITORY', 'STATE', 'COMPANY', 'SITE', '#', 'START_YR', 'Type',
                                         'Gas'], how='left', suffixes=('_old', ''))

    years = [str(i) for i in range(1978, 2051)]
    years_sigma = [year + '_sigma' for year in years]

    for year, uncert in zip(years, years_sigma):
        emissions_merged[year] = emissions_merged[year].fillna(emissions_merged[year + '_old'])
        emissions_merged[uncert] = emissions_merged[uncert].fillna(emissions_merged[uncert + '_old'])

    eth_emissions_update = emissions_merged.drop(
        columns=list(emissions_merged.columns[['_old' in i for i in emissions_merged.columns]]) + ['START_MO'])

    full_update = pd.concat((facility_emissions[facility_emissions['PRODUCT'] != 'ETHYLENE'], eth_emissions_update),
                            axis=0)

    full_update[years_sigma] = full_update[years_sigma].astype(float)

    return full_update.sort_values(list(full_update.columns[:15]))


def attribute_weighted_ethylene_to_facilities(feedstock_vals, filt_agg):
    years = list(map(str, list(range(1978, 2051))))

    print('Attributing ethylene feedstock emissions...')
    # Apply emissions to each facility
    blank = feedstock_vals[feedstock_vals.columns[:7]]
    blank.columns = list(blank.columns.droplevel(1))
    conversions = filt_agg.columns[['allocation' in name for name in filt_agg.columns]]

    for conversion in tqdm(conversions):
        fs_ems = filt_agg[conversion]
        each_conv = pd.DataFrame()
        for year in years:
            df = feedstock_vals[year]
            for fs in df.columns[1:]:
                df[fs] = df[fs] * fs_ems.loc[fs]
            yearly = blank.copy()
            yearly['Year'] = year
            yearly[conversion] = np.sum(df[df.columns[1:]].values, axis=1)
            each_conv = pd.concat((each_conv, yearly), axis=0)
        conv_emissions = pd.concat((blank, each_conv.pivot(columns=['Year'], values=conversion)), axis=1)
        conv_emissions['conversion'] = conversion
        if conversion != conversions[0]:
            ethylene_ems = pd.concat((ethylene_ems, conv_emissions),
                                     axis=0)  # .merge(each_conv, on=list(each_conv.columns[:8]), how='left')
        else:
            ethylene_ems = conv_emissions.copy()

    # ethylene_conv = ethylene_ems.copy()
    # ethylene_conv = ethylene_conv[['mass' in i for i in ethylene_conv['conversion']]]
    # ethylene_conv['conversion'] = [i.replace(', mass allocation ','_').replace('_factor','') for i in ethylene_conv['conversion']]
    # ethylene_conv.columns = [i.replace(', mass allocation ','_').replace('_factor','') for i in ethylene_conv.columns]

    return ethylene_ems


def calculate_ethylene_feedstock_emissions(conv_factors, feedstock_types,
                                           exclusion_column='ihs_cradle-to-out-gate CO2e_20a,  allocation factor'):
    eth_conv = conv_factors[conv_factors['Product'] == 'ETHYLENE'].reset_index(drop=True)
    feedstock_emissions = eth_conv.merge(feedstock_types, on='ihs_match', how='left')

    feedstock_emissions[exclusion_column] = feedstock_emissions[exclusion_column].astype(float)
    keep_match_locs = \
    feedstock_emissions.groupby('Feedstock').apply(exclude_outliers).drop(columns=['Feedstock']).reset_index()[
        'level_1']
    keep_matches = eth_conv.loc[keep_match_locs]
    keep_rows = feedstock_emissions['ihs_match'].isin(keep_matches['ihs_match'])
    feedstock_emissions = feedstock_emissions[keep_rows]

    filt_agg = feedstock_emissions.drop(
        columns=['Product', 'Product group', 'Product type', 'ei_match', 'cm_match', 'ihs_match']).groupby(
        ['Feedstock']).mean()

    ## Get technology uncertainty by taking stdev
    stdevs = feedstock_emissions[
        ['Feedstock'] + [i for i in feedstock_emissions.columns if 'ihs' in i and 'sigma' not in i]].drop(
        columns='ihs_match').groupby(['Feedstock']).agg(np.std)

    # Keep largest uncertainty between technologies and others
    years_sigma = [i for i in feedstock_emissions.columns if 'ihs' in i and 'sigma' in i]
    filt_agg[years_sigma] = np.abs((stdevs.fillna(0).values - filt_agg.fillna(0)[years_sigma].values) / 2) + np.minimum(
        stdevs.fillna(0).values, filt_agg.fillna(0)[years_sigma].values)

    return filt_agg


def get_ethylene_feedstock_vals(facility_production, feedstocks):
    # Get emissions for each feedstock
    years = list(map(str, list(range(1978, 2051))))
    # Ethylene feedstocks

    feedstocks_orig = feedstocks.copy()
    feedstock_types = pd.read_csv(input_path + 'extra_inputs/feedstock_type.csv')

    feedstocks.columns = ['_'.join(col).strip() for col in feedstocks.columns.values]

    eth_prod = facility_production[facility_production['PRODUCT'] == 'ETHYLENE']
    feedstock_matches = feedstocks.merge(eth_prod, how='left', left_on=list(feedstocks.columns[:6]),
                                         right_on=['COUNTRY/TERRITORY', 'STATE', 'COMPANY', 'SITE', '#', 'START_YR'])

    capacity_cols = [i for i in feedstock_matches.columns if 'CAPACITY' in str(i)]

    for col, year in zip(capacity_cols, years):
        feedstock_matches[col] = feedstock_matches[year]

    feedstock_matches.drop(columns=list(facility_production.columns), inplace=True)
    feedstock_matches.columns = feedstocks_orig.columns

    feedstock_vals = feedstock_matches.copy()
    for year in years:
        df = feedstock_vals[year]
        df['CAPACITY'] = df['CAPACITY'].apply(lambda x: re.sub("[^0-9.]", "0", str(x))).astype(float)
        df[df.columns[1:]] = df[df.columns[1:]].multiply(df['CAPACITY'] / 100, axis='index')
        feedstock_vals[year] = df

    return feedstock_vals


def aggregate_facility_emissions(facility_emissions):
    years = [str(i) for i in range(1978, 2051)]
    years_sigma = [year + '_sigma' for year in years]

    print('Aggregating facility emissions...')
    facility_emissions[facility_emissions.columns[:13]] = facility_emissions[facility_emissions.columns[:13]].fillna(
        'n.a.')
    facility_emissions[years] = facility_emissions[years].astype(float)
    facility_emissions[years_sigma] = facility_emissions[years_sigma].astype(float)
    # Take mean of possible emissions given different possible technologies for each facility
    aggregated_emissions = facility_emissions.drop(columns='PROCESS').groupby(
        list(facility_emissions.columns[:13]) + ['Gas', 'Type']).agg(np.mean)
    print('Facility mean done')

    ## Get technology uncertainty by taking stdev
    stdevs = facility_emissions[list(facility_emissions.columns[:13]) + ['Gas', 'Type'] + years].groupby(
        list(facility_emissions.columns[:13]) + ['Gas', 'Type']).agg(np.std)
    print('Facility stdev done')

    # Keep largest uncertainty between technologies and others
    aggregated_emissions[years_sigma] = np.maximum(stdevs.fillna(0).values,
                                                   aggregated_emissions.fillna(0)[years_sigma].values)

    aggregated_emissions = aggregated_emissions.reset_index()
    aggregated_emissions[aggregated_emissions.columns[:15]] = aggregated_emissions[
        aggregated_emissions.columns[:15]].astype(str)

    return aggregated_emissions


def calculate_facility_emissions(facility_conversion_orig, dbs=None, names=None, emission_val_cols=None, output_path='', current_group_name=''):
    if dbs is None:
        dbs = ['combined_', 'ihs_cradle-to-out-gate ', 'Feedstock ', 'Primary chemicals ', 'Intermediates ',
               'Indirect Utilities ', 'Direct Utilities ', 'Direct Process ', 'Electricity ', 'Thermoplastics ',
               'N-fertilisers ', 'Solvents, additives & explosives ', 'Thermosets, fibre & elastomers ',
               'Other downstream ']
    if names is None:
        names = ['EI & CM', 'IHS CtOG', 'Feedstock', 'Primary chemicals', 'Intermediates', 'Indirect Utilities',
                 'Direct Utilities', 'Direct Process', 'Electricity', 'Thermoplastics', 'N-fertilisers',
                 'Solvents, additives & explosives', 'Thermosets, fibre & elastomers', 'Other downstream']
    if emission_val_cols is None:
        emission_val_cols = ['CO2e_20a', 'CO2e_100a', 'CO2e_500a', 'Carbon dioxide', 'Carbon monoxide', 'Chloroform',
                             'Dinitrogen monoxide', 'Ethane', 'Methane', 'Nitric oxide', 'Nitrogen fluoride',
                             'Perfluoropentane', 'Sulfur hexafluoride']

    for i, emission_val_col in tqdm(enumerate(emission_val_cols)):
        emission_val_col = [emission_val_col]
        emission_val_col_sigma = [col + '_sigma' for col in emission_val_col]
        facility_conversion = facility_conversion_orig.copy()

        for column, col_sigma in tqdm(zip(emission_val_col, emission_val_col_sigma)):

            if (len(facility_conversion['cm_' + column + '_conv_factor'].dropna()) == 0) or (
                    sum(facility_conversion['cm_' + column + '_conv_factor'].dropna()) == 0):
                facility_conversion['combined_' + column] = facility_conversion['ei_' + column + '_conv_factor']
                facility_conversion['combined_' + col_sigma] = facility_conversion[
                    'ei_' + column + '_conv_factor_sigma']
            else:
                facility_conversion['combined_' + column] = np.nanmean(
                    [facility_conversion['ei_' + column + '_conv_factor'],
                     facility_conversion['cm_' + column + '_conv_factor']], axis=0)
                facility_conversion['combined_' + col_sigma] = np.nanmean(
                    [facility_conversion['ei_' + column + '_conv_factor_sigma'],
                     facility_conversion['cm_' + column + '_conv_factor_sigma']], axis=0)

        facility_conversion = facility_conversion[
            facility_conversion.columns[['ei' not in col and 'cm' not in col for col in facility_conversion.columns]]]

        facility_conversion.rename(columns={'ihs_match': 'PROCESS'}, inplace=True)

        facility_conversion.columns = [i.replace(',  allocation ', '_').replace('_factor', '') for i in
                                       facility_conversion.columns]

        # Create base dataframe to use
        years = [str(i) for i in range(1978, 2051)]
        years_sigma = [year + '_sigma' for year in years]
        base_columns = ['PRODUCT', 'COUNTRY/TERRITORY', 'STATE', 'COMPANY', 'SITE', '#',
                        'ROUTE', 'TECHNOLOGY', 'LICENSOR', 'START_YR', 'COMPLEX', 'LATITUDE', 'LONGITUDE',
                        'PROCESS'] + years + years_sigma
        base_df = facility_conversion[base_columns]

        facility_emissions = pd.DataFrame()
        for db, name in tqdm(zip(dbs, names)):
            for gas in tqdm(emission_val_col):
                if db + gas in facility_conversion.columns:
                    df = base_df.copy()
                    df[years] = df[years].multiply(facility_conversion[db + gas], axis='index')
                    ## Incorrect error propagation here
                    df[years_sigma] = df[years_sigma].multiply(facility_conversion[db + gas + '_sigma'], axis='index')
                    df['Gas'] = gas
                    df['Type'] = name
                    facility_emissions = pd.concat((facility_emissions, df), axis=0)

        os.mkdir(output_path + 'temp_facility_ems') if not os.path.exists(output_path + 'temp_facility_ems') else None
        facility_emissions.to_parquet(
            output_path + 'temp_facility_ems/facilityEmissions_' + current_group_name + '_' + str(i) + '.parquet')

    if len(emission_val_cols) > 1:
        facility_emissions = pd.concat([pd.read_parquet(
            output_path + 'temp_facility_ems/facilityEmissions_' + current_group_name + '_' + str(i) + '.parquet') for i
                                        in range(len(emission_val_cols))], axis=0)

    return facility_emissions


def import_ifa(file_path):
    # name_conversions = {
    #     'NH3': 'AMMONIA',
    #     'AN': 'AMMONIUM NITRATE',
    #     'Ammonium nitrate (33.5-0-0) granulated': 'AMMONIUM NITRATE',
    #     'AS': 'AMMONIUM SULPHATE',
    #     'CAN': 'CALCIUM AMMONIUM NITRATE',
    #     'Calcium ammonium nitrate (27-0-0)': 'CALCIUM AMMONIUM NITRATE',
    #     'Urea (46-0-0)': 'UREA'
    # }

    ifa_ihs_matches = {
        'AMMONIA': 'AMMONIA',
        'AMMONIUM NITRATE': 'AMMONIUM NITRATE',  # 'AMMONIUM NITRATE FERTILIZER',
        'AMMONIUM SULPHATE': 'AMMONIUM SULPHATE',  # 'HYDROXYLAMMONIUM SULFATE',
        'CALCIUM AMMONIUM NITRATE': 'AMMONIUM NITRATE',  # 'AMMONIUM NITRATE FERTILIZER',
        'UREA': 'UREA',  # 'UREA, AGRICULTURAL GRADE'
    }
    ifa_production = pd.read_csv(file_path)
    # ifa_production['PRODUCT'] = ifa_production['PRODUCT'].replace(name_conversions)
    ifa_production.rename(columns={'Country': 'COUNTRY/TERRITORY'}, inplace=True)
    ifa_production['Conv_name'] = ifa_production['PRODUCT'].replace(ifa_ihs_matches)

    return ifa_production


## Add IFA production
def add_ifa_production(facility_conversion, ifa_production, conv_factors, ammonia_processes):
    ## Exclude outliers
    if 'AMMONIA' in conv_factors['Product'].unique():
        conv_factors = conv_factors.merge(ammonia_processes, on='ihs_match', how='left')
        left_cols, right_cols = ['Conv_name', 'ROUTE'], ['Product', 'Type']
    else:
        left_cols, right_cols = ['Conv_name'], ['Product']
    poss_ifa = ifa_production.merge(conv_factors, left_on=left_cols, right_on=right_cols, how='left').drop(
        columns=['Conv_name', 'Product'])
    cols = ['PRODUCT', 'ROUTE']
    ifa_years = [str(i) for i in range(1978, 2051)]
    # keep_rows = poss_ifa[cols+['ihs_match', 'ihs_cradle-to-out-gate CO2e_20a,  allocation factor']].groupby(cols).apply(exclude_outliers)
    # filt_ifa = poss_ifa.iloc[list(keep_rows.index.get_level_values(1))].reset_index(drop=True)
    ifa_conversion = poss_ifa[
        ['COUNTRY/TERRITORY'] + ifa_years + cols + [i + '_sigma' for i in ifa_years] + ['ihs_match']]

    facility_conversion = pd.concat((facility_conversion, ifa_conversion))

    return facility_conversion


def merge_matching_processes(facility_production, poss_processes,
                             determinant_column='ihs_cradle-to-out-gate CO2e_20a,  allocation factor'):
    """Function for getting unexcluded processes from IHS and ICIS matches"""
    cols = ['PRODUCT', 'ROUTE', 'TECHNOLOGY', 'LICENSOR']
    poss_processes[determinant_column] = poss_processes[determinant_column].astype(float)
    keep_rows = poss_processes[cols + ['ihs_match', determinant_column]].groupby(cols).apply(exclude_outliers)
    filt_processes = poss_processes.iloc[list(keep_rows.index.get_level_values(4))].reset_index(drop=True)
    icis_ihs_matches = filt_processes[['ihs_match'] + cols]

    facility_conversion = facility_production.merge(icis_ihs_matches, on=cols, how='left')
    return facility_conversion


# define a function to exclude outliers
def exclude_outliers(group, col='ihs_cradle-to-out-gate CO2e_20a,  allocation factor'):
    if len(group) > 3:  # only exclude outliers if the group has more than 3 rows
        mean = np.mean(group[col])
        std = np.std(group[col])
        max_distance = std  # maximum distance from the mean to be considered an outlier
        distances = np.abs(group[col] - mean)  # calculate distances of each value to the mean
        filtered_group = group[distances <= max_distance]  # keep only values within the maximum distance

        if len(filtered_group) < 3:  # if less than 3 rows remain, take the 3 closest to the mean
            group['dist'] = np.abs(group[col] - mean)
            closest_rows = group.nsmallest(3, 'dist', keep='all')

            return closest_rows.drop(columns=['dist'])
        else:
            return filtered_group
    else:
        return group


# ## Weight ammonia conversion factor
# def weight_ammonia_conversion_factor(conv_factors, ammonia_processes, sr_percentage=0.8):
#     # IEA’s ammonia roadmap only considered three key processes for producing ammonia synthesis gas: steam reforming of natural gas (with the highest share of production, 63%), coal gasification (34%), and partial oxidation/steam reforming of oil feedstocks such as naphtha, LPG and fuel oil (2%).
#     # Use only these for now and for scenario include green ammonia.
#
#     grouped_amm = conv_factors[conv_factors['Product']=='AMMONIA'].merge(ammonia_processes, on='ihs_match').drop(columns=['Product', 'Product type', 'Product group', 'ei_match', 'cm_match', 'ihs_match']).groupby('Type').mean()
#     amm_weighted = (1-sr_percentage)*grouped_amm.iloc[0, :]+sr_percentage*grouped_amm.iloc[1, :]
#
#     amm_df = pd.DataFrame(amm_weighted).transpose().drop(columns=['Total']).astype(float)
#     amm_df['Product'], amm_df['ihs_match'] = 'AMMONIA', 'WEIGHTED AMMONIA'
#     amm_df.index = [3000]
#     conv_factors = pd.concat((conv_factors[conv_factors['Product']!='AMMONIA'], amm_df))
#
#     return conv_factors

def add_ifa_conv_factors(combined_factors, ifa_factors):
    ifa_factors['Product'] = ifa_factors['Product'].str.upper()
    ifa_matches = pd.read_csv(input_path + 'extra_inputs/ifa_matches.csv')
    ifa_factors = ifa_factors.merge(ifa_matches.dropna(), on='Product', how='right').drop(columns='Product').rename(
        columns={'Match': 'Product'})

    combined_factors = combined_factors.merge(ifa_factors, on='Product', how='left')

    return combined_factors


def calculate_implied_emissions_factors(allocation_df, material_emissions, emission_val_cols, suffixes=['']):
    """Function for calculating implied emissions factors from allocated emissions"""

    total_allocation = allocation_df[allocation_df['Type'] == 'Product']
    IHS_emissions = total_allocation[
        ['Target/Process', 'Product'] + [i for i in total_allocation.columns if 'allocated' in i]]
    IHS_emissions.rename(columns={'Target/Process': 'ihs_match'}, inplace=True)

    group_names = ['Total', 'Raw Material', 'Primary chemicals', 'Intermediates', 'Direct Utilities',
                   'Indirect Utilities', 'Direct Process', 'Electricity', 'Thermoplastics', 'N-fertilisers',
                   'Solvents, additives & explosives', 'Thermosets, fibre & elastomers', 'Other downstream']
    output_names = ['ihs_cradle-to-out-gate ', 'Feedstock ', 'Primary chemicals ', 'Intermediates ',
                    'Direct Utilities ', 'Indirect Utilities ', 'Direct Process ', 'Electricity ', 'Thermoplastics ',
                    'N-fertilisers ', 'Solvents, additives & explosives ', 'Thermosets, fibre & elastomers ',
                    'Other downstream ']

    for suffix in suffixes:
        for group_name, output_name in zip(group_names, output_names):
            catch = [IHS_emissions.rename(columns={
                group_name + ' allocated ' + col + suffix: output_name + col + ', ' + suffix[1:] + ' allocation factor',
                group_name + ' allocated ' + col + '_sigma' + suffix: output_name + col + ', ' + suffix[
                                                                                                 1:] + ' allocation sigma'},
                                          inplace=True) for col in emission_val_cols]

    mats = material_emissions.rename(columns={'Product': 'Target', 'Source/Object': 'Product'}).drop_duplicates(
        subset=['Product']).drop(
        columns=['Code', 'Data Version', 'Type', 'Target/Process', 'Research Year', 'Country/Reg', 'Target', 'Value',
                 'Value unit', 'Value_sigma', 'Capacity unit', 'MeasType', 'Provenance'])

    combined_factors = mats.sort_values('Product').reset_index(drop=True)
    combined_factors = combined_factors.merge(IHS_emissions, on='Product', how='outer')
    combined_factors = combined_factors.dropna(subset=combined_factors.columns[1:], how='all').reset_index(drop=True)
    return combined_factors


def filter_df_for_type(df, types, type_col):
    return df[[i in types for i in df[type_col]]]


def calculate_type_emissions(materials_df, product_df, emission_types: list, group_name: str, emissions_cols: list,
                             emissions_cols_sigma: list, emission_val_cols: list, product_ratio_col, product_value_col,
                             emission_type_col='Type'):
    # Impose lists
    if type(emission_type_col) is str:
        emission_type_col = [emission_type_col]
        emission_types = [emission_types]
    if type(emission_types) is str:
        emission_types = [emission_types]

    # Sum for groups
    grouped_df, grouped_df_sigma = materials_df.copy(), materials_df.copy()
    for emission_type_list, col in zip(emission_types, emission_type_col):
        if grouped_df.empty is False:
            grouped_df = filter_df_for_type(grouped_df, emission_type_list, col)

    group_ems = np.sum(grouped_df[emissions_cols])
    group_ems_sigma = np.sum(grouped_df[emissions_cols_sigma])

    # Loop through each value/gas column
    for val_column, gas_column, val_column_sigma, gas_column_sigma in zip(emission_val_cols, emissions_cols,
                                                                          [col + '_sigma' for col in emission_val_cols],
                                                                          emissions_cols_sigma):
        # Allocate emissions for value and uncertainty
        product_df[group_name + ' allocated ' + val_column] = group_ems[gas_column] * product_df[product_ratio_col]
        product_df[group_name + ' allocated ' + val_column_sigma] = uncertainty_propagation('mult',
                                                                                            group_ems[gas_column],
                                                                                            group_ems_sigma[
                                                                                                gas_column_sigma],
                                                                                            product_df[
                                                                                                product_ratio_col],
                                                                                            product_df[
                                                                                                product_ratio_col + '_sigma'],
                                                                                            z=product_df[
                                                                                                group_name + ' allocated ' + val_column])

        # Calculate emissions intensity for values and uncertainty
        product_df[group_name + ' unit emission intensity ' + val_column] = product_df[
                                                                                group_name + ' allocated ' + val_column] / \
                                                                            product_df[product_value_col]
        product_df[group_name + ' unit emission intensity ' + val_column_sigma] = uncertainty_propagation('mult',
                                                                                                          product_df[
                                                                                                              group_name + ' allocated ' + val_column].values,
                                                                                                          product_df[
                                                                                                              group_name + ' allocated ' + val_column_sigma].values,
                                                                                                          product_df[
                                                                                                              product_value_col].values,
                                                                                                          product_df[
                                                                                                              product_value_col + '_sigma'].values,
                                                                                                          z=product_df[
                                                                                                              group_name + ' unit emission intensity ' + val_column])

    return product_df


def allocate_emissions(df: pd.DataFrame, emission_val_cols: list, mass_to_other_convs=False,
                       mass_to_other_uncertainty=0.01, value_col='Mass, kg', ratio_col='Mass ratio'):
    # Get inputs to products
    df_ins = df[df['Type'] != 'By-Product']
    for column in emission_val_cols:
        if len(df_ins['cm_' + column + '_cradle-to-gate'].dropna()) == 0:
            df_ins['combined_' + column] = df_ins['ei_' + column + '_cradle-to-gate']
            df_ins['combined_' + column + '_sigma'] = df_ins['ei_' + column + '_cradle-to-gate_sigma']
        else:
            df_ins['combined_' + column] = np.nanmean(
                [df_ins['ei_' + column + '_cradle-to-gate'], df_ins['cm_' + column + '_cradle-to-gate']], axis=0)
            df_ins['combined_' + column + '_sigma'] = np.nanmean(
                [df_ins['ei_' + column + '_cradle-to-gate_sigma'], df_ins['cm_' + column + '_cradle-to-gate_sigma']],
                axis=0)

    combined_cols = ['combined_' + column for column in emission_val_cols]
    combined_cols_sigma = ['combined_' + column + '_sigma' for column in emission_val_cols]

    allocation = pd.DataFrame()

    # Loop through each process
    for code in df['Code'].unique():

        # Get by-products and mass ratios
        temp = df[df['Code'] == code][
            ['Code', 'Data Version', 'Source/Object', 'Type', 'Target/Process', 'Research Year', 'Country/Reg',
             'Product', 'Value', 'Value unit', 'Value_sigma', 'Capacity unit', 'MeasType', 'Provenance']]
        a = temp.iloc[0]
        a['Source/Object'], a['Type'], a['Value'], a['Value unit'], a['Value_sigma'] = a['Product'], 'Product', float(
            1), 'kg/kg', float(0)
        a = pd.DataFrame(a.values.reshape(1, -1), columns=a.index)
        temp = temp[temp['Type'] == 'By-Product']
        temp = pd.concat([temp, a], axis=0)

        # Convert values to energy if conversion exists in mass_to_enrgy_convs file
        if mass_to_other_convs is not False:
            # value_col, ratio_col, unit = 'Energy, MJ', 'Energy ratio', 'MJ'
            temp = temp.merge(mass_to_other_convs, how='left', left_on='Source/Object',
                              right_on=mass_to_other_convs['Product'].str.upper())
            if temp['Conversion'].isnull().values.any():
                continue
            else:
                temp[value_col] = temp['Conversion'] * abs(temp['Value'])
                temp[value_col + '_sigma'] = uncertainty_propagation('mult', abs(temp['Value']), temp['Value_sigma'],
                                                                     temp['Conversion'],
                                                                     mass_to_other_uncertainty * temp['Conversion'],
                                                                     z=temp[value_col])

        else:
            # value_col, ratio_col, unit = 'Mass, kg', 'Mass ratio', 'kg'
            temp[value_col] = abs(temp['Value'])
            temp[value_col + '_sigma'] = temp['Value_sigma']

        # Get ratio of product vs all products+by-products
        temp[ratio_col] = temp[value_col] / sum(temp[value_col])
        temp[ratio_col + '_sigma'] = uncertainty_propagation('mult', temp[value_col], temp[value_col + '_sigma'],
                                                             sum(temp[value_col]), sum(temp[value_col + '_sigma']),
                                                             z=temp[ratio_col])
        ### -> Assumption of adding uncertainties together for sum(temp[value_col+'_sigma'])

        # Get process emissions & allocate
        used_mats = df_ins[df_ins['Code'] == code]

        unique_types = list(used_mats['Type'].unique())
        group_names = ['Total'] + unique_types
        types_lists = [unique_types] + unique_types

        for group_name, types in zip(group_names, types_lists):
            temp = calculate_type_emissions(used_mats, temp, types, group_name, combined_cols, combined_cols_sigma,
                                            emission_val_cols, ratio_col, value_col)

        temp = calculate_type_emissions(used_mats, temp, [['Indirect Utilities'], ['ELECTRICITY']], 'Electricity',
                                        combined_cols, combined_cols_sigma, emission_val_cols, ratio_col, value_col,
                                        emission_type_col=['Type', 'Source/Object'])

        # Identify missing material emissions
        temp['Missing raw materials (>1% mass)'] = str(used_mats[(used_mats['Type'] == 'Raw Material') & (
                    str(used_mats[combined_cols[0]]) == 'nan') & (used_mats['Value'] > 0.01 * np.sum(
            used_mats['Value']))]['Source/Object'].tolist())

        # Identify missing utility emissions
        temp['Missing utilities'] = str(
            used_mats[(used_mats['Type'] == 'Utilities') & (str(used_mats[combined_cols[0]]) == 'nan')][
                'Source/Object'].tolist())

        # Add current product to allocation list
        allocation = pd.concat([allocation, temp], axis=0)
    # return used_mats
    return allocation


def convert_feedstocks_to_intermediates(material_emissions, product_group):
    upstream_prods = \
    product_group[[i in ['Primary chemicals', 'Intermediates'] for i in product_group['Product group']]]['Product']
    equivs_dict = dict(zip(product_group['Product'], product_group['Product group']))
    material_emissions.loc[(material_emissions['Type'] == 'Raw Material') & (
        material_emissions['Source/Object'].isin(upstream_prods)), 'Type'] = 'Diff'
    material_emissions.loc[(material_emissions['Type'] == 'Diff'), 'Type'] = \
    material_emissions.loc[(material_emissions['Type'] == 'Diff')]['Source/Object'].replace(equivs_dict)

    return material_emissions


def get_direct_process_emissions(input_emissions, direct_emissions):
    # Add emissions for each direct process
    process_emissions = \
    input_emissions[[i in list(direct_emissions['Product']) for i in list(input_emissions['Product'])]][
        list(input_emissions.columns[:14]) + ['Product type', 'Product group']].drop_duplicates(
        subset=['Code', 'Target/Process', 'Product']).reset_index(drop=True)
    process_emissions['Type'], process_emissions['MeasType'] = 'Direct Process', 'Chemical'
    process_emissions['Source/Object'] = process_emissions['Product']
    process_emissions['Value'], process_emissions['Value_sigma'] = 1, 0
    process_emissions = process_emissions.merge(direct_emissions, on='Product', how='inner')

    # Merge with all input emissions
    output_emissions = pd.concat((input_emissions, process_emissions), axis='index').sort_values(
        by=['Product', 'Target/Process', 'Code', 'Type', 'Source/Object'])

    return output_emissions


def calc_direct_process_convs(direct_emissions, product_process_match, ei_emissions):
    # Direct process emissions matching
    # Import direct emissions and match to existing products in ihsMaterials
    emission_val_cols = list(ei_emissions.columns[3:16])
    emission_val_cols_sigma = list(ei_emissions.columns[16:])

    direct_emissions = direct_emissions[['Process'] + list(direct_emissions.columns[-5:])]
    direct_emissions['Process'] = direct_emissions['Process'].str.upper()

    direct_emissions = direct_emissions.merge(product_process_match, on='Process', how='right').dropna(
        subset=['Product']).drop(columns=['Process']).drop_duplicates(subset=['Product']).rename(
        columns={'est. CO2': 'Carbon dioxide', 'est. CH4': 'Methane', 'est. N2O': 'Nitric oxide',
                 'est. CO2e_20a': 'CO2e_20a', 'est. CO2e_100a': 'CO2e_100a'})
    direct_emissions['CO2e_500a'] = (direct_emissions['CO2e_100a'] + direct_emissions['Carbon dioxide']) / 2

    uncertainty_ratio = 0.01

    for col in emission_val_cols:
        if col in list(direct_emissions.columns):
            direct_emissions['ei_' + col + '_cradle-to-gate'] = direct_emissions[col].fillna(0).astype(float)
            direct_emissions['ei_' + col + '_cradle-to-gate_sigma'] = (
                        direct_emissions[col].astype(float) * uncertainty_ratio)
            direct_emissions.drop(columns=[col], inplace=True)
        else:
            direct_emissions['ei_' + col + '_cradle-to-gate'] = 0
            direct_emissions['ei_' + col + '_cradle-to-gate_sigma'] = 0

    return direct_emissions


def get_direct_energy_emissions(material_emissions, direct_utl_conv, ei_emissions):
    """Function for adding direct energy emissions to material emissions"""
    # Add direct emissions for each utility
    emission_val_cols = list(ei_emissions.columns[3:16])
    emission_val_cols_sigma = list(ei_emissions.columns[16:])

    direct_utl_ems = material_emissions[material_emissions['Type'] == 'Indirect Utilities'][
        list(material_emissions.columns)[:14] + ['Product type', 'Product group']]
    direct_utl_ems['Type'] = 'Direct Utilities'
    direct_utils = direct_utl_ems.merge(direct_utl_conv, left_on='Source/Object', right_on='Source', how='left').rename(
        columns={'Source': 'ei_match'})

    for col in emission_val_cols + emission_val_cols_sigma + ['Value', 'Value_sigma']:
        direct_utils[col] = direct_utils[col].astype(float)

    for gas in emission_val_cols:
        direct_utils['ei_' + gas + '_cradle-to-gate'] = direct_utils['Value'] * direct_utils[gas]
        direct_utils['ei_' + gas + '_cradle-to-gate_sigma'] = uncertainty_propagation('mult', direct_utils['Value'],
                                                                                      direct_utils['Value_sigma'],
                                                                                      direct_utils[gas],
                                                                                      direct_utils[gas + '_sigma'],
                                                                                      z=direct_utils[
                                                                                          'ei_' + gas + '_cradle-to-gate'])
        direct_utils['ei_' + gas + '_conv_factor'] = direct_utils[gas]
        direct_utils['ei_' + gas + '_conv_factor_sigma'] = direct_utils[gas + '_sigma']

    direct_utils.drop(columns=emission_val_cols + emission_val_cols_sigma, inplace=True)

    # Merge with material emissions
    input_emissions = pd.concat((material_emissions, direct_utils), axis='index').sort_values(
        by=['Product', 'Target/Process', 'Code', 'Type', 'Source/Object'])

    return input_emissions


def get_upstream_emissions(input_mats, ei_emissions, cm_emissions, match_list_ei, match_list_cm):
    """Function for getting upstream emissions including feedstocks and indirect energy usage from input materials"""
    # Match equivalent emissions to materials
    emission_val_cols = list(ei_emissions.columns[3:16])
    emission_val_cols_sigma = list(ei_emissions.columns[16:])

    # EI matching
    material_emissions, upt_list = assign_emissions(input_mats, ei_emissions, 'Source/Object', 'Source',
                                                    match_list=match_list_ei, db_name='ei',
                                                    emission_val_cols=emission_val_cols,
                                                    emission_val_cols_sigma=emission_val_cols_sigma)

    # match_list_ei = pd.concat((match_list_ei, upt_list)).drop_duplicates(subset=['IHS'], keep='last')

    # CM matching
    material_emissions, upt_list = assign_emissions(material_emissions, cm_emissions, 'Source/Object', 'Source',
                                                    match_list=match_list_cm, db_name='cm',
                                                    emission_val_cols=emission_val_cols,
                                                    emission_val_cols_sigma=emission_val_cols_sigma)

    # match_list_cm = pd.concat((match_list_cm, upt_list)).drop_duplicates(subset=['IHS'], keep='last')

    # # Combine match lists
    # all_matches = match_list_ei[['IHS','ei']]
    # all_matches['cm'] = match_list_cm['cm']
    # all_matches.sort_values('IHS').reset_index(drop=True).to_csv(match_list_path, index=False)

    # Create materials emissions
    material_emissions = material_emissions.drop_duplicates(subset=['Code', 'Source/Object']).reset_index(drop=True)

    return material_emissions


def filter_rows(df: pd.DataFrame, column: str, item: str, exact: bool = True):
    """Function for finding best match for input item in a df column"""
    # If exact match enforced
    if exact:
        return df[df[column].str.lower() == item.lower()]

    # If item is in string but not entire string
    else:
        return df[[item in row for row in df[column].str.lower()]]


def uncertainty_propagation(calc: str, x: float, dx: float, y: float = 1, dy: float = 0, z: float = 1,
                            propagation_type: str = 'simple') -> float:
    """Function for propagating uncertainty through calculations"""
    # Multiplication
    if calc == 'mult':
        xdiv = np.divide(dx, x, out=np.zeros_like(dx), where=x != 0)
        ydiv = np.divide(dy, y, out=np.zeros_like(dy), where=y != 0)
        if propagation_type == 'simple':
            return (xdiv + ydiv) * z
        elif propagation_type == 'stdev':
            return np.sqrt(pow(xdiv, 2) + pow(ydiv, 2)) * z
        else:
            Exception('Specified propagation_type not recognised.')

    # Addition
    elif calc == 'add':
        if propagation_type == 'simple':
            return abs(dx) + abs(dy)
        elif propagation_type == 'stdev':
            return np.sqrt(pow(dx, 2) + pow(dy, 2))
        else:
            Exception('Specified propagation_type not recognised.')
    else:
        Exception('Please specify calc of propagation')


def assign_emissions(df: pd.DataFrame, emissions_df: pd.DataFrame, product_col: str, emissions_col: str,
                     product_val_col: str = 'Value', emission_val_cols: list = None,
                     emission_val_cols_sigma: list = None, match_list=None, db_name: str = 'db',
                     production_unit_conv: float = 1) -> (pd.DataFrame, pd.DataFrame):
    """These function assigns appropriate emissions values from EcoInvent or Carbonminds to products or materials in IHS given a pre-determined match from file or finding the best matches available"""

    # Create values if none exist
    if match_list is None:
        match_list = {}
    if emission_val_cols is None:
        emission_val_cols = ['Cradle-to-gate']
    if emission_val_cols_sigma is None:
        emission_val_cols_sigma = ['Cradle-to-gate_sigma']
    product_val_col_sigma = product_val_col + '_sigma'

    # Create columns to receive emissions, match name, emissions conversion factor
    val_col, match_name_col, conv_factor_col = pd.DataFrame(columns=emission_val_cols), [], pd.DataFrame(
        columns=emission_val_cols)
    # Columns for uncertainties of above
    val_col_sigma, conv_factor_col_sigma = pd.DataFrame(columns=emission_val_cols_sigma), pd.DataFrame(
        columns=emission_val_cols_sigma)

    # Create match dictionary from appropriate match dataframe column
    length = len(emission_val_cols + emission_val_cols_sigma)
    if isinstance(match_list, pd.DataFrame) and db_name in match_list.columns:
        match_list = dict(zip(match_list['IHS'], match_list[db_name]))

    # Loop through rows in assignment dataframe
    for row_num, row in tqdm(enumerate(df.iloc())):

        # Check match_list for match
        if row[product_col].lower() in match_list.keys():

            # If already defined as no match in db
            if str(match_list[row[product_col].lower()]) == '0':
                correspondence = pd.DataFrame()
                emission_val, name = pd.DataFrame(np.array([np.nan] * length).reshape(1, length),
                                                  columns=emission_val_cols + emission_val_cols_sigma), np.nan
            # If match has corresponding db value
            else:
                correspondence = filter_rows(emissions_df, emissions_col, match_list[row[product_col].lower()])
                emission_val = correspondence[emission_val_cols + emission_val_cols_sigma]
                name = correspondence.iloc[0][emissions_col]

        # If no match yet assigned
        else:
            # Find correspondence in emissions dataframe
            correspondence = filter_rows(emissions_df, emissions_col, row[product_col].lower())  # Exact matching

            if len(correspondence) == 0:  # No exact match -> Try name contained within a match
                correspondence = filter_rows(emissions_df, emissions_col, row[product_col].lower(), exact=False)

                if len(correspondence) > 1:  # If multiple inexact matches
                    take = input('Enter number of best match for ' + row[product_col].lower() + ':\n' + str(
                        correspondence[emissions_col]) + '\n Type n to skip')  # Ask user for best match
                    if take != 'n':
                        correspondence = correspondence[correspondence[emissions_col] == correspondence.loc[int(take)][
                            emissions_col]]  # Take best match
                    else:
                        correspondence = pd.DataFrame()  # If none correspond then empty correspondence

            if len(correspondence) == 0:  # No exact match -> Try match contained within name
                matching = emissions_df[[i in row[product_col].lower() for i in
                                         emissions_df[emissions_col]]]  # Emission string contained in row matching

                if len(matching) > 0:  # If multiple matches
                    correspondence = matching.iloc[np.argmax(
                        [len(i) for i in matching[emissions_col]])]  # Take greatest length of match if multiple
                    emission_val = correspondence[emission_val_cols + emission_val_cols_sigma]
                    name = correspondence[emissions_col]

                else:
                    emission_val, name = pd.DataFrame(np.array([np.nan] * length).reshape(1, length),
                                                      columns=emission_val_cols + emission_val_cols_sigma), np.nan  # If no matches identified

            else:
                emission_val = correspondence[emission_val_cols + emission_val_cols_sigma]
                name = correspondence[emissions_col].values[0]

            # Add match to match_list
            if len(correspondence) != 0:
                if isinstance(correspondence, pd.DataFrame):
                    match_list.update({row[product_col].lower(): correspondence.iloc[0]['Source']})
                else:
                    match_list.update({row[product_col].lower(): correspondence['Source']})
            else:
                match_list.update({row[product_col].lower(): 0})
            del correspondence

        # Add matching values to dataframe
        val_col = pd.concat((val_col, row[product_val_col] * production_unit_conv * emission_val[emission_val_cols]))

        # Calculate implied uncertainties and add to dataframe
        val_col_sigma = pd.concat((val_col_sigma,
                                   pd.DataFrame(
                                       uncertainty_propagation('mult', row[product_val_col], row[product_val_col_sigma],
                                                               emission_val[emission_val_cols].values,
                                                               emission_val[emission_val_cols_sigma].values, z=(
                                                       row[product_val_col] * production_unit_conv * emission_val[
                                                   emission_val_cols]).values) * production_unit_conv,
                                       columns=val_col_sigma.columns)))

        # Add other parameters to parameter lists
        match_name_col += [name]
        conv_factor_col = pd.concat((conv_factor_col, emission_val[emission_val_cols]))
        conv_factor_col_sigma = pd.concat((conv_factor_col_sigma, emission_val[emission_val_cols_sigma]))

    df[db_name + '_match'] = match_name_col
    for column, sigma_col in zip(emission_val_cols, emission_val_cols_sigma):
        df[db_name + '_' + column + '_cradle-to-gate'] = val_col[column].values
        df[db_name + '_' + column + '_cradle-to-gate_sigma'] = val_col_sigma[sigma_col].values
        df[db_name + '_' + column + '_conv_factor'] = conv_factor_col[column].values
        df[db_name + '_' + column + '_conv_factor_sigma'] = conv_factor_col_sigma[sigma_col].values

    return df, pd.DataFrame.from_dict(match_list, orient='index').reset_index().rename(
        columns={'index': 'IHS', 0: db_name})