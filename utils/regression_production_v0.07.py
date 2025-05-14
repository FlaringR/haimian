#############################################trend only#########################
import os
import datetime
import json
import gc
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LogisticRegression
import ray

warnings.filterwarnings('ignore')

def logistic_reg_fit_binary(X, y, sample_weight, max_iter=400):
    impute = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
    scaler = StandardScaler()
    model = LogisticRegression(fit_intercept=True, max_iter=max_iter)
    pipeline = make_pipeline(impute, scaler, model)
    pipeline.fit(X, y, logisticregression__sample_weight=sample_weight)
    return pipeline

def logistic_reg_predict_binary(pipeline, X):
    proba = pipeline.predict_proba(X)
    return proba[:,1]

def custom_qcut(factor, bins=20):
    return pd.qcut(factor, bins, retbins=True, labels=False, duplicates='drop')

def weighted_mean(x, weights):
            return np.dot(x, weights) / weights.sum()

def get_factor_data3(factor_table, folder='/home/haimian/mnt_225/factor/300750.SZ/300750.SZ/', file0='/home/haimian/mnt_225/factor/300750.SZ/data_20240813_trend_duo.ftr', file1='/home/haimian/mnt_225/factor/300750.SZ/open_all.ftr', open_all='open_all_duo_124_in_long_trend_1515'):
    factor_list = factor_table.factor.unique().tolist()
    files = [f+'.ftr' for f in factor_list]
    df13 = []
    for f in files:
        df_ = pd.read_feather(os.path.join(folder, f)).set_index('Time').fillna(0)
        df13.append(df_)
    df13 = pd.concat(df13, axis=1)
    df0 = pd.read_feather(file0, columns=['Time', 'y60_duo','y120_duo','y180_duo', 'close_position_point_11_ret_skip_60', 'close_position_point_11_ret_skip_120','close_position_point_11_ret_skip_180']).set_index('Time')
    df1 = pd.read_feather(file1, columns=[open_all,'Time']).set_index('Time')
    return pd.concat([df0, df1, df13], axis=1)


def prepare_data(factor_table, y_tags, folder, file0, file1, open_all='open_all', fee1 = 0.002, fee2=0.0015): 
    factors = get_factor_data3(factor_table, folder=folder, file0=file0, file1=file1, open_all=open_all)
    # factors = factors.loc[factors[select_class]]
    factors = factors.loc[factors[open_all]]
    # factors.drop(select_class, axis=1, inplace=True)
    factors.drop(open_all, axis=1, inplace=True)
    factors['date'] = factors.index.date
    for y_tag in y_tags:
        factors = factors.loc[factors[y_tag].notna()]
        factors.loc[factors.index<='2023-08-28', y_tag+'_miusFee'] = factors.loc[factors.index<='2023-08-28', y_tag] - fee1
        factors.loc[factors.index>'2023-08-28', y_tag+'_miusFee'] = factors.loc[factors.index>'2023-08-28', y_tag] - fee2
        factors.loc[factors[y_tag+'_miusFee'] <= 0, y_tag  + '_clf'] = 0
        factors.loc[factors[y_tag+'_miusFee'] > 0, y_tag  + '_clf'] = 1
    return factors.sort_index()


def is_os_split(dates, train_set_size=63, test_set_size=1, move_steps=1):
    split_list = []
    n = len(dates)
    i = 0
    while (i + train_set_size) < n:
        trainset = dates[i:(i + train_set_size)]
        testset = dates[(i + train_set_size): (i + train_set_size + test_set_size)]
        split_list.append((trainset, testset))
        i = i + move_steps
    return split_list

def is_os_split_v3(index, train_set_size=1000):
    split_list = []
    dates = sorted(pd.Series(index.date).unique().tolist())
    train_dates = []
    pdIndex = pd.Series(index.date, index=index).sort_index()
    # print(pdIndex)
    i = 0
    for date in dates[:-1]:
        i +=1
        train_dates.append(date)
        set1 = pdIndex.loc[pdIndex.isin(train_dates[1:])]
        if len(set1) < train_set_size:
            continue
        if len(set1) > train_set_size:
            train_dates.pop(0)
        train_set = train_dates.copy()
        split_list.append((train_set,  [dates[i]]))
    return split_list

def rankic(factor, y_f):
    return abs(pd.concat([factor.loc[factor.notna()], y_f], axis=1).corr(method='spearman').iloc[0,1])

def rankic_v2(factor, y_f, perc=0.4):
    df = pd.concat([factor.loc[factor.notna()], y_f], axis=1)
    n = int(len(df) * perc)
    name = factor.name
    sorted_df = df.sort_values(by=[name], ascending=True)

    if name.startswith('AskPrice1_diff_1_over_LastPrice_dayk_nega_perc_ewm'):
        return - sorted_df.iloc[:n].corr(method='spearman').iloc[0,1]
    else:
        return sorted_df.iloc[-n:].corr(method='spearman').iloc[0,1]



def main(factor_table, substring='', code = '300750.SZ', train_days=63,  y_tags = ['y60_duo', 'y120_duo',  'y180_duo', 'close_position_point_n480_118_kong_ret'], \
    data_version='122', data_date='20240813', trade_type='duo', open_all='open_all_duo_124_in_long_trend_1515',\
        select_class='p_trend_perc_124_4_type_3', corr_threshold_lev1=0.05, corr_threshold_lev2=0.0, nmin=3, min_trainning_sample_points=1000, num_cpus=4, model_params_folder='/home/haimian/mnt_215/strategy/regression/regression_results_long/300750.SZ/model_params/', freq_str='ma_8'):

    if trade_type =='duo':
        direction = 'long'
    elif trade_type == 'duo_ping':
        direction = 'long_close'
    elif trade_type == 'kong':
        direction = 'short'
    elif trade_type == 'kong_ping':
        direction = 'short_close'

    factor_data_path = '/home/haimian/mnt_225/factor/{}/{}/'.format(code, code)
    tag_data_path = '/home/haimian/mnt_225/factor/{}/data_{}_trend_{}.ftr'.format(code, data_date, trade_type)
    open_all_path = '/home/haimian/mnt_225/factor/{}/open_all.ftr'.format(code)
    

    factors = prepare_data(factor_table, y_tags, folder=factor_data_path, file0=tag_data_path, file1=open_all_path, open_all=open_all)

    is_os_dates0 = is_os_split(sorted(factors['date'].unique().tolist()), train_set_size=train_days, test_set_size=1, move_steps=1)
    is_os_dates0_dict = dict()
    for x, y in is_os_dates0:
        is_os_dates0_dict[y[0]] = x

    is_os_dates1 = is_os_split_v3(factors.index, train_set_size=min_trainning_sample_points)
    is_os_dates1_dict = dict()
    for x, y in is_os_dates1:
        is_os_dates1_dict[y[0]] = x
    df_is_os =pd.concat([pd.Series(is_os_dates1_dict), pd.Series(is_os_dates0_dict)], axis=1).dropna(axis=0)

    def return_max_is_dates(x):
        x0 = x[0]
        x1 = x[1]
        if len(x0)>=len(x1):
            return x0
        else:
            return x1
    df_is_os = df_is_os.apply(return_max_is_dates, axis=1)

    is_os_dates = []
    for i in range(len(df_is_os)):
        is_os_dates.append((df_is_os.iloc[i], [df_is_os.index[i]]))


    factor_cols = [x for x in factors.columns.tolist() if x in factor_table.factor.tolist()]
    y_tags = [x+ '_clf' for x in y_tags]

    ray.shutdown()
    ray.init(address='auto')

    factors_id = ray.put(factors)

    @ray.remote(num_cpus=num_cpus)
    def ray_exec(x, y,  factors):
        date = y[0]
        print(date)
        if date < datetime.date(2020, 6, 10):
        # if date < datetime.date(2023, 11, 16):
            return pd.DataFrame()
        similar_dates = x
        train = factors.loc[(factors['date'].isin(similar_dates))]
        # train = prepare_sample(factor_table, train)
        dict1 = pd.Series(train['date'].unique()).to_dict()
        n_dict = len(dict1)
        train_weights_dict = dict([(value, (key+1)/n_dict) for key, value in dict1.items()])
        weights = train['date'].map(train_weights_dict)
        weights.name = 'weights'
        train = pd.concat([train, weights], axis=1)
        

        test = factors.loc[date.strftime('%Y%m%d')]
        del factors
        gc.collect()
        
        
        if len(test) == 0:
            print('{}: no test data'.format(date))
            return pd.DataFrame()
        try:
            proba_df_test = pd.DataFrame()
            params = dict()
            for y_tag in y_tags:
                print(y_tag)
                params[y_tag] = dict()
                # remove the factors that have low rankics
                corr = train[factor_cols].apply(rankic, y_f=train[y_tag[:-4]])
                # # corr = calc_weigthed_corr(train[factor_cols], train[y_tag[:-4]], train['weights'], nbins=nbins)
                factor_cols_selct = corr.loc[corr>corr_threshold_lev1].sort_values(ascending=False).index.tolist()

                corr2 = train[factor_cols].apply(rankic_v2, y_f=train[y_tag[:-4]])
                factor_cols_selct2 = corr2.loc[corr2>corr_threshold_lev2].sort_values(ascending=False).index.tolist()

                factor_cols_selct = [value for value in factor_cols_selct if value in factor_cols_selct2]

                indicators = []
                for value in factor_cols_selct:
                    if value.startswith('net_act_vol_mean_5_uniform_ccc30_ewm') or  value.startswith('net_vol_over_max_of_vol_and_volume_mean_5_ewm') or\
                        value.startswith('order_book_uniform_ccc30_ewm') or value.startswith('AskPrice1_ge_AskPrice2_ccc30_ewm'):
                        indicators.append(True)
                    else:
                        indicators.append(False)

                # remove the factors that have high inter-factor correlations
                # mu_selected = calc_weighted_mean_at_last_level(train[factor_cols_selct], train[y_tag[:-4]], train['weights'], nbins=nbins)
                # retain one of them which has the highest rankic
                # factor_cols_selct = eliminate_high_correlation(train[factor_cols_selct], corr.loc[factor_cols_selct], threshold=corr_threshold_lev2)[:nmax]
                # factor_cols_selct = factor_cols

                # if len(factor_cols_selct) <= nmin:
                #     proba_series_test = pd.Series(index=test.index, name='class_{}_'.format(direction)+y_tag, dtype='float64')
                # else:
                if sum(indicators) < 4:
                    proba_series_test = pd.Series(index=test.index, name='class_{}_'.format(direction)+y_tag, dtype='float64')
                else:
                    pipe = logistic_reg_fit_binary(train[factor_cols_selct], train[y_tag], train['weights'])
                    proba_test = logistic_reg_predict_binary(pipe, test[factor_cols_selct])
                    proba_series_test = pd.Series(proba_test, index=test.index, name='class_{}_'.format(direction)+y_tag)


                proba_df_test = pd.concat([proba_df_test, proba_series_test, test[y_tag[:-4]]], axis=1)
                # proba_df_test.loc[:, 'factor_cols_selected_{}'.format(y_tag[:-4])] = ','.join(factor_cols_selct)
            # param_file = os.path.join(model_params_folder, "{}_{}.json".format(date.strftime('%Y%m%d'), freq_str))
            # with open(param_file, "w") as file:
            #     json.dump(params, file)
            proba_df_test = pd.concat([proba_df_test, test[['close_position_point_11_ret_skip_60', 'close_position_point_11_ret_skip_120','close_position_point_11_ret_skip_180']]], axis=1)
            return proba_df_test
        except Exception as err:
            print('failed on {}:    {}'.format(date, err))
            return  pd.DataFrame()
    resid = [ray_exec.remote(x, y, factors_id) for x, y in is_os_dates]
    res = ray.get(resid)   

    ray.shutdown()
    if len(res) > 0:
        return pd.concat(res)
    else:
        return pd.DataFrame()

if __name__=="__main__":
    for file in ["config_production_v0.07_duo_regression.json"]:
        with open(file, "r") as jsonfile:
            param = json.load(jsonfile)
        for code in param['code']:
            print(code)
            output_folder='/home/haimian/mnt_215/strategy/regression/regression_results_long/{}/'.format(code)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            model_params_folder='/home/haimian/mnt_215/strategy/regression/regression_results_long/{}/model_params/'.format(code)
            factor_table_all=pd.read_feather('/home/haimian/mnt_225/factor/{}/factor_{}{}_{}.ftr'.format(code, param['data_version'], param['substring'], param['trade_type']))

            for o, c in zip(param['open_all'], param['select_class']):
                factor_table = factor_table_all.loc[(factor_table_all['type'] == c)] 
                experiment_list = ['ma_1','ma_2','ma_4','ma_8','ma_16','ma_32','ma_64','ma_128']
                factor_selc = [
                    [
                        'net_act_vol_mean_5_uniform_ccc30_ewm',
                        'net_vol_over_order_mixed_0_ewm', 
                        'net_vol_over_max_of_vol_and_volume_mean_5_ewm',
                        'order_book_uniform_ccc30_ewm',
                        'AskPrice1_diff_1_over_LastPrice_dayk_nega_perc_ewm',
                        'AskPrice1_ge_AskPrice2_ccc30_ewm'
                    ], 
                    # [
                    #     'net_vol_ewm_over_net_vol_ma_d_100',
                    #     'net_act_vol_mean_5_uniform_ccc30_ewm',
                    #     'net_vol_over_max_of_vol_and_volume_mean_5_ewm',
                    #     'net_vol_over_order_mixed_0_ewm', 
                    #     'order_book_uniform_ccc30_ewm',
                    #     'AskPrice1_diff_1_over_LastPrice_dayk_nega_perc_ewm',
                    #     'AskPrice1_ge_AskPrice2_ccc30_ewm'
                    # ]       
                ]

                for lev1 in param["corr_threshold_lev1"]:
                    for lev2 in param["corr_threshold_lev2"]:
                        for nmin in param['nfactor_min']:
                            for i in range(len(experiment_list)):
                                for j in range(len(factor_selc)):
                                    factor_table_i = factor_table.loc[factor_table.factor.str.endswith(experiment_list[i])]
                                    factor_table_ij = factor_table_i.loc[factor_table_i.factor_type.isin(factor_selc[j])]
                                    version_duo = '{}_{}days_{}_{}_experiment{}_group{}_corr1{}_corr2{}_nmin{}'.format(param['version'], str(param['train_days']),  o, c, experiment_list[i], j, round(lev1*100),round(lev2*100), int(nmin))
                                    output_file = "regression_results_{}.ftr".format(version_duo)
                                    if True:
                                        result = main(factor_table_ij, substring=param['substring'], code = code, train_days=param['train_days'], y_tags=param['y_tags'], data_version=param['data_version'],\
                                            data_date=param['data_date'], trade_type=param['trade_type'], open_all=o,\
                                                    select_class=c, corr_threshold_lev1=lev1, corr_threshold_lev2=lev2, nmin=nmin, min_trainning_sample_points=1000, num_cpus=param['num_cpus'])
                                        if len(result) > 0:
                                            result.reset_index().to_feather(os.path.join(output_folder, output_file))
                                        else:
                                            print('{}:  the result is empty'.format(code))