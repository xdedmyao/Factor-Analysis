"""
@ Author: 薛定谔的猫
@ Date: 2025.07.16
"""
import pandas as pd
import numpy as np
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.tsa.stattools import acf, adfuller
import statsmodels.api as sm
from tqdm import tqdm 
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')

DATES_PATH = '../data/dates/trading_dates.parquet'
DAILY_PATH = '../data/data_daily/daily.parquet'
BASIC_INFO_PATH = '../data/stock_info/stock_info.parquet'
INDUSTRY_INFO = '../data/industry/sw_industry_info.parquet'
SHARE_PATH = '../data/shares/shares.parquet'
    
def trading_date_offset(date:str, offset=1):
    """
    给定日期, 找到距离该日期前offset个交易日的日期(默认偏移为1)

    Args:
    - date: 给定的日期
    - offset: 距离给定日期的偏移天数(>0)

    Return:
    1. 若给定日期偏移后超出trading_dates的数据范围(2020-2025年), 返回 'Index Outside the Range'
    2. 正常情况, 返回偏移offset个交易日的日期

    """
    # 确保偏移日大于0
    if offset < 0:
        return 'Offset should be positive'
    
    trading_dates = pd.read_parquet(DATES_PATH)
    valid_dates = trading_dates[trading_dates.trade_status == 1].trade_date.values.tolist()

    # 找到给定日期在数据中的位置
    for index in range(len(valid_dates)):
        if date <= valid_dates[index]:
            date_index = index
            break
        
    # 偏移后日期不在数据中
    if date_index - offset < 0:
        return 'Index Outside the Range'
    
    return valid_dates[date_index - offset]

def trading_dates(start_date:str, end_date:str):
    """
    给定起止日期，返回中间的所有的交易日列表(包含端点)
    
    Args:

    - start_date
    - end_date
        
    Return:
    
    - List
    """
    if start_date > end_date:
        return None
    
    trading_dates = pd.read_parquet(DATES_PATH)
    valid_dates = trading_dates[trading_dates.trade_status == 1].trade_date.values.tolist()
    
     # 找到给定日期在数据中的位置
    for index in range(len(valid_dates)):
        if start_date <= valid_dates[index]:
            start_index = index
            break
        
    for index in range(len(valid_dates)):
        if end_date < valid_dates[index]:
            end_index = index
            break
    
    return valid_dates[start_index:end_index]

def get_daily_data(start_date:str, end_date:str):
    """
    获取给定日期区间的股票日线数据(剔除上市不满一年、涨跌停的股票)
    
    Args:
    - start_date
    - end_date
    
    Return:
    - pd.DataFrame(Columns: stock_code, date, open, close, high, low, volume, money, mcap, prev_close)
    
    """
    if start_date > end_date:
        return None
    
    def filter_stocks(df:pd.DataFrame):
        """
        剔除上市不满一年的股票
        
        Args:
        - df: pd.DataFrame(Columns: stock_code,exchange,date,open,close,high,low,volume,money)
                
        Return:
        - pd.DataFrame(Columns: stock_code,exchange,date,open,close,high,low,volume,money)
        """
        stock_info = pd.read_parquet(BASIC_INFO_PATH)

        # 转换成日期格式
        df.date = pd.to_datetime(df.date)
        stock_info.list_date = pd.to_datetime(stock_info.list_date)

        # 计算上市天数
        merge_df = pd.merge(df, stock_info, on = ['stock_code'],how = 'left')
        merge_df['days_listed'] = (merge_df['date'] - merge_df['list_date']).dt.days

        # 剔除上市不满365天
        df_filtered = merge_df[merge_df['days_listed'] >= 365]

        return df_filtered.drop(columns=['short_name']).reset_index(drop=True)

    result = pd.read_parquet(DAILY_PATH)
    # 剔除停牌的股票

    result = result[result['trading_status'] == 0].reset_index(drop=True)
    result = result[(result.date >= start_date) & (result.date <= end_date)].reset_index(drop = True)
    return filter_stocks(result)

def get_shares(start_date:str, end_date:str):
    """
    获取指定时间短内所有A股流通股本数据
    
    Args:
   
    - start_date
    - end_date
    
    Return:
    - pd.DataFrame(包含stock_code, date, shares)
    
    """
    result = pd.read_parquet(SHARE_PATH)
    return result[(result.date >= start_date) & (result.date <= end_date)].reset_index(drop=True)

def Factor_Processing(df, factor_name, neutralization=False, zscore=True, cut_extreme = True):

    def factor_mad_cut_extreme(df:pd.DataFrame, factor_name="factor_value", k=3):
        """
        因子横截面MAD去极值模块
        
        MAD:计算因子值偏离中位数的绝对偏差的中位数，作为衡量离散程度的指标
        因子值被限制在 [median(X) - k * MAD, median(X) + k * MAD]
        
        Args:
        - df: pd.Dataframe(Columns: stock_code, date, factor_value)  
        - k: 默认为3
            
        Return:
        - pd.DataFrame(Columns: stock_code, date, factor_value)

        """
        def mad_winsorize_group(group, k=k):
            median_val = group.median()
            mad = np.median(np.abs(group - median_val))
            if mad == 0:
                return group
            return np.clip(group, median_val - k * mad, median_val + k * mad)
                        
        df[factor_name] = df.groupby('date')[factor_name].transform(lambda x:mad_winsorize_group(x))

        return df

    def factor_neutralize(df: pd.DataFrame, factor_name="factor_value"):
        """
        对每个交易日的横截面数据进行市值+行业中性化处理，返回中性化后的因子残差。
        
        Args:
        - df: 输入 DataFrame, 需包含股票代码、日期、因子值
        
        Return:
        - pd.DataFrame: 中性化后的因子值，索引与原始数据一致
        """
        def cross_sectional_neutralize(df: pd.DataFrame, 
                                    factor_col: str = 'factor_value',
                                    date_col: str = 'date',
                                    code_col: str = 'stock_code',
                                    cap_col: str = 'float_mcap',
                                    industry_col: str = 'industry_code',
                                    log_cap: bool = True) -> pd.DataFrame:
            df = df.copy()
            df = df.dropna(subset=[factor_col, cap_col, industry_col])  # 删除缺失值
            df[industry_col] = df[industry_col].astype(str)
            
            neutralized = []
            for date, group in tqdm(df.groupby(date_col), desc='Processing Date:'):
                if len(group) < 2:
                    neutralized.append(pd.Series(np.nan, index=group.index))
                    continue

                # 市值中性化
                y = group[factor_col].astype(float)
                X = group[[cap_col]]
                
                if log_cap:
                    if (X[cap_col] <= 0).any():
                        print(f"Warning: Non-positive market cap on {date}")
                        neutralized.append(pd.Series(np.nan, index=group.index))
                        continue
                    X[cap_col] = np.log(X[cap_col])
                
                # 行业中性化
                # 添加行业哑变量
                if group[industry_col].nunique() > 1:
                    industry_dummies = pd.get_dummies(group[industry_col], prefix='industry', drop_first=True)
                    X = pd.concat([X, industry_dummies], axis=1)
                
                X = sm.add_constant(X)
                X = X.astype(float)
                try:
                    model = sm.OLS(y, X).fit()
                    resid = model.resid
                except Exception as e:
                    # print(f"Error on {date}: {e}")
                    resid = pd.Series(np.nan, index=group.index)
                
                neutralized.append(resid)
            
            neutralized_series = pd.concat(neutralized).reindex(df.index)
            df[factor_col] = neutralized_series
            return df[[code_col, date_col, factor_col]]
        
        # 合并行业数据
        industry_df = pd.read_parquet(INDUSTRY_INFO)[['stock_code', 'industry_code']]
        merged_df = pd.merge(df, industry_df, on=['stock_code'], how='left')
        
        # 中性化
        neutralized_df = cross_sectional_neutralize(merged_df, factor_col = factor_name)
        
        return neutralized_df

    def factor_zscore(df:pd.DataFrame, fill_method='zero', factor_name="factor_value"):
        """
        因子横截面 Z-Score 标准化
        
        Args:
        - df : DataFrame
            必须包含 ['stock_code', 'date', 'factor_value'] 三列
            
        fill_method : str, 可选 ('zero', 'nan', 'remove')
            处理标准差为0时的策略：
            - 'zero'：将 zscore 设为0（默认）
            - 'nan' ：保留原始 NaN
            - 'remove'：删除该日期数据
            
        Return:
        - pd.DataFrame -['stock_code', 'date', factor_name] 三列
        """
        
        required_cols = {'stock_code', 'date', factor_name}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")
        
        df = df.copy()
        
        # 按日期分组计算统计量
        grouped = df.groupby('date')[factor_name]
        mean = grouped.transform('mean')
        std = grouped.transform('std')
        
        # 处理零标准差
        if fill_method == 'zero':
            std = std.replace(0, 1)  # 当std=0时，分子为0，结果强制为0
        elif fill_method == 'nan':
            std = std.replace(0, np.nan)
        elif fill_method == 'remove':
            valid_dates = grouped.filter(lambda x: x.std() != 0)['date']
            df = df[df['date'].isin(valid_dates)]
            mean = df.groupby('date')[factor_name].transform('mean')
            std = df.groupby('date')[factor_name].transform('std')
        else:
            raise ValueError(f"Invalid fill_method: {fill_method}. Use 'zero', 'nan', or 'remove'")
        
        # 计算 Z-Score
        df[factor_name] = (df[factor_name] - mean) / std
        
        return df[['stock_code', 'date', factor_name]]
    
    print('Process: 因子预处理...')
    result_df = df.copy()
    if cut_extreme == True:
        print('Process: 因子截面MAD去极值...')
        result_df = factor_mad_cut_extreme(result_df, factor_name = factor_name)
    
    if neutralization == True:
        print('Process: 因子中性化')
        result_df = factor_neutralize(result_df, factor_name = factor_name)

    if zscore == True:
        print('Process: 因子Zscore标准化...')
        result_df = factor_zscore(result_df, factor_name = factor_name)

    return result_df
    
def factor_rank_autocorrelation(df:pd.DataFrame):
    """
    因子自相关性分析模块
    
    Args:
    - df: pd.DataFrame()(columns: stock_code, date, factor_value)
        
    Return:
    - 中间输出: 通过平稳性检验的股票的个数
    - pd.DataFrame(): 滞后为1天、5天、10天的自相关系数
        
    """
    
    """ 计算平均因子rank自相关结果 """
    
    print('正在进行自相关性分析...')
    df['date'] = pd.to_datetime(df['date'])

    # 确保数据按日期排序
    df = df.sort_values('date')

    # 计算每日横截面因子值的rank
    df['factor_rank'] = df.groupby('date')['factor_value'].rank(method='average')

    # 对每只股票的rank序列计算自相关(n=1,5,10)
    def safe_acf(series, nlags=1):
        if len(series) < 3:  # 最少需要3个点计算可靠自相关
            return np.nan
        try:
            return acf(series, nlags=nlags, fft=False)[nlags]  
        except:
            return np.nan
        
    acf_by_stock_1 = df.groupby('stock_code')['factor_rank'].apply(safe_acf, nlags=1)
    acf_by_stock_5 = df.groupby('stock_code')['factor_rank'].apply(safe_acf, nlags=5)
    acf_by_stock_10 = df.groupby('stock_code')['factor_rank'].apply(safe_acf, nlags=10)
    
    # 计算平均自相关（忽略NaN）
    mean_acf_rank_1 = acf_by_stock_1.dropna().mean()
    mean_acf_rank_5 = acf_by_stock_5.dropna().mean()
    mean_acf_rank_10 = acf_by_stock_10.dropna().mean()

    result = pd.DataFrame(index =['Mean Factor Rank Autocorrelation'], columns = ['1D','5D','10D'])
    result['1D'] = mean_acf_rank_1.round(3)
    result['5D'] = mean_acf_rank_5.round(3)
    result['10D'] = mean_acf_rank_10.round(3)
    
    """ ADF检验 """
    
    # 检查NaN值
    nan_count = df['factor_value'].isna().sum()
    if nan_count > 0:
        print("存在NaN值，建议处理（删除或填充）。当前代码将跳过含NaN的股票。")

    df['factor_value'] = df['factor_value'].fillna(0)
    
    # 定义ADF检验函数
    def run_adf(series, min_length=3):
        
        # 检查序列长度和是否为常数
        if len(series) < min_length or series.isna().any():
            return np.nan
        if series.nunique() == 1:  # 常数序列
            return np.nan
        
        # 运行ADF检验
        result = adfuller(series)
        return result[1]

    # 对每只股票的因子值序列进行ADF检验
    adf_results = df.groupby('stock_code')['factor_value'].apply(run_adf).reset_index()

    adf_results.columns = ['stock_code','p_value']
    adf_results['stationary'] = adf_results['p_value'].apply(
        lambda p: '平稳 (p < 0.05)' if p < 0.05 else '可能非平稳 (p ≥ 0.05)' if pd.notna(p) else '无法检验'
    )
    
    # 计算平稳股票比例
    valid_results = adf_results.dropna(subset=['p_value'])
    stationary_count = (valid_results['p_value'] < 0.05).sum()
    total_valid = len(valid_results)
    print(f"\n平稳股票数量: {stationary_count}/{total_valid} ({stationary_count/total_valid:.2%})")
    return result

def factor_backtest(df: pd.DataFrame, 
                    factor_name: str,  
                    start_date: str,  
                    end_date: str,    
                    lag_days: int = 2, 
                    direction: int = 1,     
                    group: int = 5,         
                    neutralization = False, 
                    zscore = True,
                    cut_extreme = True): 
    """
    因子分组回测函数，合并所有图表到一张大图
    
    Args:
    - df: pd.DataFrame, 包含 stock_code, date, factor_value 三列
    - factor_name: str, 因子名称
    - start_date: str,  开始日期
    - end_date: str,    结束日期
    - lag_days: int,    使用滞后多少天的收益率
    - direction: int,   因子方向(默认为正向)
    - group: int,       分组数量，默认 5
    - neutralization: bool, 市值，行业中性化选项
    - zscore: bool,     标准化选项
    - cut_extreme: bool,截面MAD去极值选项
            
    Return:
    - result: pd.DataFrame
    - cumulative_return: pd.DataFrame
    - ic: pd.Series

    """
    if factor_name not in df.columns:
        print('Factor name does not exist')
        return
    
##################################### 因子预处理 ################################################
    df = Factor_Processing(df, 
                        factor_name = factor_name, 
                        neutralization=neutralization, 
                        zscore=zscore,
                        cut_extreme=cut_extreme)

##################################### 合并收益率信息 ########################################################
    ret = get_daily_data(start_date, end_date)[['stock_code', 'date', 'close']]
    ret['return'] = ret.groupby('stock_code')['close'].pct_change().fillna(0)
    ret = ret.drop(columns=['close'])
    ret['date'] = pd.to_datetime(ret['date'])
    merge_df = pd.merge(ret, df, on=['stock_code', 'date'], how='left')

##################################### 因子回测计算函数 #######################################################
    
    def backtest(df, factor_name, direction, num_groups=5, lag_days=1):
        df = df.sort_values(['date', 'stock_code']).copy()

        # 计算滞后收益率
        df['return_adjusted'] = df.groupby('stock_code')['return'].shift(-lag_days)
        df = df.dropna(subset=['return_adjusted'])
        df['group'] = df.groupby('date')[factor_name].transform(
            lambda x: pd.qcut(x, num_groups, labels=False, duplicates='drop') + 1)
        
        # 计算 IC
        ic = df.groupby('date').apply(
            lambda x: x[factor_name].corr(x['return_adjusted'], method='pearson')
        )

        # 分组计算收益
        group_returns = df.groupby(['date', 'group'])['return_adjusted'].mean().unstack()
        all_groups = list(range(1, num_groups + 1))
        group_returns = group_returns.reindex(columns=all_groups).fillna(0)
        group_returns['IC'] = ic
        
        # 若 IC 为负则反转分组标签
        if direction < 0:
            reversed_cols = list(range(num_groups, 0, -1))
            group_returns.columns = reversed_cols
            
        # 指标计算
        result = pd.DataFrame(index=['value'])
        result['IC'] = round(ic.mean(), 3)
        result['ICIR'] = round(abs(ic.mean() / ic.std() if ic.std() != 0 else np.nan), 3)
        
        # 对冲收益计算
        group_returns['long_short'] = group_returns[num_groups] - group_returns[1]
        
        # 日期滞后调整
        cumulative_returns = group_returns.cumsum()
        cu_returns = (1 + group_returns).cumprod()
        cumulative_returns.dropna(inplace=True)

        # 年化收益率计算
        annual_return_long = group_returns[num_groups].mean() * 252
        annual_return_long_short = group_returns.long_short.mean() * 252
        result['Long Annual Return'] = round(annual_return_long, 3)
        result['Long Max Drawdown'] = round(((cu_returns[num_groups].cummax() - cu_returns[num_groups]) / cu_returns[num_groups].cummax()).max(), 3)
        result['Long Sharpe'] = round((group_returns[num_groups].mean() * 252 - 0.1) / (group_returns[num_groups].std() * np.sqrt(252)), 3)
        result['Short Max Drawdown'] = round(((cu_returns[1].cummax() - cu_returns[1]) / cu_returns[1].cummax()).max(), 3)
        result['Short Sharpe'] = round((group_returns[1].mean() * 252 - 0.1) / (group_returns[1].std() * np.sqrt(252)), 3)
        result['LS Annual Return'] = round(annual_return_long_short, 3)
        result['LS Max Drawdawn'] = round(((cu_returns['long_short'].cummax() - cu_returns['long_short']) / cu_returns['long_short'].cummax()).max(), 3)
        result['LS Sharpe'] = round((group_returns.long_short.mean() * 252 - 0.1) / (group_returns.long_short.std() * np.sqrt(252)), 3)
        return result, group_returns, ic

##################################### 绘图函数 ###############################################################
    
    def plot_all_charts(result, group_returns, ic, df, factor_name, num_groups, width=15, height=23):
        
        # 创建 4x2 网格的 GridSpec
        fig = plt.figure(figsize=(width, height))
        gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1.5, 1, 1])
        fig.suptitle(f'Factor Analysis: {factor_name}', fontsize=20, weight='bold')

##################################### 子图 1: 分组因子均值和日度收益(第一行左) ################################################

        print('Process: 绘制分层曲线...')
        ax1 = fig.add_subplot(gs[0, 0])
        group_daily_mean_ret = [group_returns[i].mean() for i in range(1, num_groups+1)]
        index = [i for i in range(1, num_groups+1)]
        
        bars = ax1.bar(index, group_daily_mean_ret, 
                       color="#CCD7DB", 
                       width=0.6,
                       edgecolor='black')
        
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.4f}',
                     ha='center', va='bottom')
        ax1.axhline(0, color="#000000", linestyle='-', linewidth=1.2, zorder=0)
        ax1.set_xticks(index)
        ax1.set_title('Group Mean Return(Daily)', fontsize=15)
        ax1.set_xlabel('Group', fontsize=12)
        ax1.set_ylabel('Mean Return', fontsize=12)
        ax1.grid(axis='y', alpha=0.4, linestyle='--')

################################### 子图 2: 股票数量统计(第一行右) ################################################
        
        ax2 = fig.add_subplot(gs[0, 1])
        x_ticks_interval = len(group_returns.index) // 20
        df_dropna = df.copy().dropna()
        num_stocks = df_dropna.groupby('date')['stock_code'].count()
        ax2.plot(num_stocks, )
        ax2.set_xticks(num_stocks.index[::x_ticks_interval])
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_title('Number of Stocks', fontsize=15)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Counts', fontsize=12)
        ax2.grid(axis='y', alpha=0.4, linestyle='--')

################################## 子图 3: Cumulative Returns (第二行全部) ################################################

        ax3 = fig.add_subplot(gs[1, :])
        if num_groups <= 5:
            colors = ['#4CAB6D','#74D571','#62B9B6','#4885B8','#C02626']
        else:
            cmap = plt.cm.get_cmap('coolwarm')
            values = np.linspace(0, 1, num_groups)
            colors = [cmap(value) for value in values]
        
        cumulative_returns = group_returns.cumsum()
        for g in range(1, num_groups + 1):
            ax3.plot(cumulative_returns.index, 
                     cumulative_returns[g], 
                     label=f'Group {g}',
                     color = colors[g-1],
                     alpha=1,
                     linewidth=1.8)
            
        ax3.set_xticks(cumulative_returns.index[::x_ticks_interval])
        ax3.tick_params(axis='x', rotation=45)
        ax3.set_title(f"Cumulative Returns (Groups={num_groups})", fontsize=15)
        ax3.set_ylabel("Cumulative Return", fontsize=12)
        ax3.legend(loc='upper left', frameon=False, fontsize=8)
        ax3.grid(True, linestyle=':', alpha=0.7)

################################### 子图 4: 多空表现图 (第三行左) ################################################

        ax4 = fig.add_subplot(gs[2, 0])
        ax4.plot(cumulative_returns['long_short'], 
                 label='Long-Short Portfolio', 
                 color="#000000", 
                 linewidth=1.5,
                 linestyle='-.',
                 marker='o', 
                 markersize=4,
                 markevery=30)
        ax4.axhline(0, color='#2F4F4F', linestyle='--', linewidth=1.2, zorder=0)
        ax4.fill_between(cumulative_returns.index, 
                         cumulative_returns['long_short'], 
                         0,
                         where=(cumulative_returns['long_short'] >= 0),
                         color="#9CA2A8", 
                         alpha=0.15,
                         interpolate=True)
        ax4.fill_between(cumulative_returns.index, 
                         cumulative_returns['long_short'], 
                         0,
                         where=(cumulative_returns['long_short'] < 0),
                         color='#9CA2A8', 
                         alpha=0.15,
                         interpolate=True)
        ax4.plot(cumulative_returns[num_groups], 
                 label='Long Portfolio', 
                 color="#C02626", 
                 linewidth=1,
                 linestyle='-')
        ax4.plot(cumulative_returns[1], 
                 label='Short Portfolio', 
                 color="#4CAB6D",
                 linewidth=1,
                 linestyle='-.')
        ax4.set_xticks(cumulative_returns.index[::x_ticks_interval])
        ax4.tick_params(axis='x', rotation=45)
        ax4.set_title("Long-Short Performance", fontsize=15)
        ax4.set_ylabel("Cumulative Return", fontsize=12)
        ax4.legend(loc='upper left', frameon=False, fontsize=10)
        ax4.grid(True, linestyle=':', alpha=0.7)

################################## 子图 5: IC 时间序列图 (第三行右) ################################################

        print('Process:因子IC分析...')
        ax5 = fig.add_subplot(gs[2, 1])
        colors = ['#1f77b4' if v >= 0 else '#d62728' for v in ic]
        ax5.bar(ic.index, ic.values, 
                color=colors, 
                width=1)
        ax5.axhline(0, color='black', linewidth=0.8)
        ax5.axhline(ic.mean(), color='black', linestyle='--', 
                    label=f'Mean IC ({ic.mean():.2f})')
        cumulative_ic = ic.cumsum()
        ax5_twin = ax5.twinx()
        ax5_twin.plot(ic.index, cumulative_ic, 
                      color="#000000", 
                      linewidth=1.5,
                      label='Cumulative IC')
        ax5_twin.set_ylabel('Cumulative IC', fontsize=12)
        ax5_twin.grid(False)
        lines1, labels1 = ax5.get_legend_handles_labels()
        lines2, labels2 = ax5_twin.get_legend_handles_labels()
        ax5.legend(lines1 + lines2, labels1 + labels2, 
                   loc='upper right', fontsize=10)
        ax5.set_xticks(ic.index[::40])
        ax5.tick_params(axis='x', rotation=45, labelsize=10)
        ax5.set_title('IC Series with Cumulative Sum', fontsize=15)
        ax5.set_ylabel('IC Value', fontsize=12)
        ax5.grid(axis='y', linestyle='--', alpha=0.7)

################################## 子图 6: 因子衰减分析 (第四行左) ################################################

        ax6 = fig.add_subplot(gs[3, 0])
        decay_result = []
        for days in range(1, 11):
            tmp_df = df.copy()
            tmp_df['return_shifted'] = tmp_df.groupby('stock_code')['return'].shift(-days)
            tmp_df = tmp_df.dropna(subset=['return_shifted'])
            ic = tmp_df.groupby('date').apply(
                lambda x: x[factor_name].corr(x['return_shifted'], method='pearson')
            )
            decay_result.append(ic.mean())
        
        indices = list(range(1, 11))
        bars = ax6.bar(indices, decay_result, 
                       color="#CCD7DB", 
                       width=0.6,
                       edgecolor='black')
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.3f}',
                     ha='center', va='bottom')
        ax6.set_xticks(indices)
        ax6.set_title('IC Decay Analysis', fontsize=15)
        ax6.set_xlabel('Days Decay', fontsize=12)
        ax6.set_ylabel('IC Mean', fontsize=12)
        ax6.grid(axis='y', alpha=0.4, linestyle='--')

##################################### 子图 7: 换手率图 (第四行右) ################################################

        print('Process: 因子换手率分析...')

        def factor_turnover(df, factor_col=factor_name, group_num=num_groups):
            def rank_and_group(group):
                group['rank'] = pd.qcut(group[factor_col], q=group_num, labels=False, duplicates='drop')
                return group
            df = df.groupby('date').apply(rank_and_group).reset_index(drop=True)
            top_group = df[df['rank'] == 0][['date', 'stock_code']].copy()
            dates = top_group['date'].unique()
            turnover_ts = {}
            for i in range(1, len(dates)):
                prev_date = dates[i-1]
                curr_date = dates[i]
                prev_stocks = set(top_group[top_group['date'] == prev_date]['stock_code'])
                curr_stocks = set(top_group[top_group['date'] == curr_date]['stock_code'])
                turnover_stocks = prev_stocks.symmetric_difference(curr_stocks)
                if len(curr_stocks) > 0:
                    turnover_rate = len(turnover_stocks) / (2 * len(curr_stocks))
                    turnover_ts[curr_date] = turnover_rate
                else:
                    turnover_ts[curr_date] = 0.0
            turnover_ts = pd.Series(turnover_ts, name='portfolio_turnover')
            turnover_ts.index = pd.to_datetime(turnover_ts.index)
            return turnover_ts

        turnover_ts = factor_turnover(df)
        result['turnover']  = turnover_ts.mean()

        ax7 = fig.add_subplot(gs[3, 1])
        ax7.plot(turnover_ts, label=f'Turnover (Mean: {turnover_ts.mean():.2f})')
        ax7.set_xticks(turnover_ts.index[::x_ticks_interval])
        ax7.tick_params(axis='x', rotation=45)
        ax7.set_title('Long Group Turnover', fontsize=15)
        ax7.set_ylabel('Turnover Rate', fontsize=12)
        ax7.legend(loc='upper left', frameon=False, fontsize=10)
        ax7.grid(True, linestyle=':', alpha=0.7)

        # 调整布局
        print('Finished')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

        return result

#####################################  生成结果  ################################################

    # 生成分组回测结果
    print('Process:因子分层回测...')
    result, group_returns, ic = backtest(merge_df, 
                                        factor_name=factor_name, 
                                        num_groups=group, 
                                        lag_days=lag_days, 
                                        direction=direction)
    
    # 绘制所有图表
    result = plot_all_charts(result, group_returns, ic, merge_df, factor_name, group)

    # 打印绩效表格
    result_transposed = result.T.reset_index()
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(tabulate(result_transposed, tablefmt='grid', showindex=False))

    return result, group_returns, ic
    