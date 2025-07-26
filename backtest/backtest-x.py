import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import spearmanr
import seaborn as sns
import statsmodels.api as sm

def signal_generation(
        df: pd.DataFrame,
        factor_col: str,
        quantile: float,
        window: int = 42
):
    df = df.copy().sort_values('date').reset_index(drop=True)

    # 向量化滚动排名
    df['factor_quantile'] = (
        df[factor_col]
        .rolling(window, min_periods=window)
        .apply(lambda x: x.rank(pct=True).iloc[-1], raw=False)
    )

    df['signal'] = 0
    df.loc[df['factor_quantile'] > quantile, 'signal'] = 1
    df.loc[df['factor_quantile'] < (1 - quantile), 'signal'] = -1
    df['signal'] = df['signal'].shift(1)
    return df


def plot_monthly_position_distribution(df, signal_col='signal', date_col='date', title='monthly position distribution'):
    """
    绘制因子信号的月度持仓分布堆叠图（多头、空头、空仓）
    df: 包含日期和 signal 列的 DataFrame
    signal_col: 信号列名，值应为 -1, 0, 1
    date_col: 日期列名
    title: 图标题
    """

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['month'] = df[date_col].dt.to_period('M')

    # 统计每月信号比例
    pivot = df.groupby('month')[signal_col].value_counts(normalize=True).unstack().fillna(0)
    print(df['signal'].value_counts())

    # 将 -1/0/1 转为中文列名（如果存在）
    rename_map = {}
    if -1 in pivot.columns:
        rename_map[-1] = 'short'
    if 0 in pivot.columns:
        rename_map[0] = 'empty'
    if 1 in pivot.columns:
        rename_map[1] = 'long'

    pivot = pivot.rename(columns=rename_map)

    # 按显示顺序排序：多头 → 空仓 → 空头（如果存在）
    plot_columns = [col for col in ['long', 'empty', 'short'] if col in pivot.columns]
    pivot = pivot[plot_columns]

    # 将 period 转换为 timestamp 以绘图
    pivot.index = pivot.index.to_timestamp()

    # 绘图
    pivot.plot(
        kind='bar',
        stacked=True,
        figsize=(14, 6),
        color=['red', 'skyblue', 'lightgreen'],
        width=1.0
    )

    plt.title(title, fontsize=14)
    plt.ylabel("percentage")
    plt.xlabel("year/month")
    plt.legend(title='position status')
    plt.grid(axis='y', linestyle='--')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.tight_layout()
    plt.show()


def calculate_ic_ir(
    df: pd.DataFrame,
    factor_col: str,
    ret_col: str,
    window: int = None,
    plot: bool = True,
    title="IC analysis"
):
    df = df[[factor_col, ret_col]].dropna().copy()
    df[ret_col] = df[ret_col].shift(-1)  # 使用 next-day return
    df = df.dropna()

    if df.empty or df[factor_col].nunique() <= 1:
        print("无有效数据或因子全为常数")
        return None, None

    if window is None:
        # 全样本 spearman 相关
        ic = spearmanr(df[factor_col], df[ret_col])[0]
        print(f" 全样本 IC: {ic:.4f}")
        return ic, None
    else:
        # 滚动窗口
        ic_list = []
        for i in range(window, len(df)):
            sub_df = df.iloc[i - window:i]
            if sub_df[factor_col].nunique() <= 1 or sub_df[ret_col].nunique() <= 1:
                ic_list.append(np.nan)
                continue
            ic_val = spearmanr(sub_df[factor_col], sub_df[ret_col])[0]
            ic_list.append(ic_val)

        ic_series = pd.Series(ic_list, index=df.index[window:])
        ir = ic_series.mean() / ic_series.std() if ic_series.std() != 0 else np.nan

        if plot:
            plt.figure(figsize=(10, 4))
            ic_series.plot(title=f'{factor_col}Rolling IC (window={window})')
            plt.axhline(0, color='gray', linestyle='--')
            plt.grid(True)
            plt.show()

            ic_series = ic_series.dropna()
            mean_ic = ic_series.mean()
            std_ic = ic_series.std()
            ir = mean_ic / std_ic if std_ic != 0 else np.nan
            over_0_percentage = ic_series[ic_series > 0].count() / ic_series.count()

            plt.figure(figsize=(14, 8))
            plt.suptitle(title, fontsize=16)

            # 1. 时间序列图
            ax1 = plt.subplot(2, 1, 1)
            ax1.plot(ic_series.index, ic_series.values, marker='o', linestyle='-', linewidth=1, markersize=3)
            ax1.axhline(y=0, color='r', linewidth=1)
            ax1.set_title(f"{factor_col} IC_time_series", fontsize=14)
            ax1.set_ylabel("IC")
            ax1.set_xlabel("date")

            # 2. 分布图 + KDE
            ax2 = plt.subplot(2, 1, 2)
            sns.histplot(ic_series, bins=30, kde=True, color='skyblue', edgecolor='black', ax=ax2)
            ax2.axvline(x=mean_ic, color='red', linestyle='--', label=f"mean:{mean_ic:.4f}")
            ax2.set_title("IC_distribution", fontsize=14)
            ax2.set_xlabel("IC")
            ax2.set_ylabel("frequency")
            ax2.legend(loc='upper right')

            # 打印 IR
            plt.figtext(0.15, 0.02, f"IC_mean: {mean_ic:.4f}    IR: {ir:.4f}", fontsize=12)

            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            plt.show()

        print(f"{factor_col}Rolling IC 均值: {ic_series.mean():.4f}, IR: {ir:.4f}, IC std: {std_ic:.4f}, IC值大于0百分比: {over_0_percentage:.4f}")
        return ic_series, ir

def regression_t_test(
    df: pd.DataFrame,
    factor_col: str,
    return_col: str,
    factor_name: str = '',
    show_plot: bool = True
):
    """
    用线性回归 return ~ factor 测试因子是否具有预测力
    """

    data = df[[factor_col, return_col]].dropna().copy()
    X = data[factor_col]
    y = data[return_col]

    # 加常数项
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    # 打印结果
    coef = model.params[factor_col]
    t_val = model.tvalues[factor_col]
    p_val = model.pvalues[factor_col]
    r2 = model.rsquared

    print(f"\n📉 regression：{factor_name or factor_col} → {return_col}")
    print(f"β: {coef:.4f}")
    print(f"t: {t_val:.3f}, p值: {p_val:.4f}")
    print(f"R²: {r2:.4f}")

    if show_plot:
        plt.figure(figsize=(6, 4))
        sns.regplot(x=factor_col, y=return_col, data=data, ci=95,
                    line_kws={"color": "red"}, scatter_kws={'s': 8})
        plt.title(f"{factor_name or factor_col} regression\nβ={coef:.3f}, t={t_val:.2f}, p={p_val:.4f}, R²={r2:.2%}")
        plt.tight_layout()
        plt.grid(True)
        plt.show()

    return model


def backtest(
    df: pd.DataFrame,
    factor_col: str,
    window: int,
    quantile: float,
    roll_window: int,
    ret_col: str = 'returns',
    delay: int = 1,
    allow_short: bool = True,
    cost_rate: float = 0.0,
):

    df = df.sort_values('date').reset_index(drop=True).copy()

    df = signal_generation(df, factor_col=factor_col, quantile=quantile, window=window)

    warmup = roll_window + window + delay
    df = df.iloc[warmup:].reset_index(drop=True)

    # 2. 初始化仓位与成本
    df['position'] = 0.0
    df['cost'] = 0.0
    prev_signal = np.nan
    current_pos = 0.0

    for i in range(len(df)):
        sig = df.loc[i, 'signal']
        if pd.isna(sig):
            df.loc[i, 'position'] = current_pos
            continue

        if not np.isclose(sig, prev_signal, atol=1e-6):
            # signal变化 → 调仓
            if allow_short:
                current_pos = sig
            else:
                current_pos = max(sig, 0)
            prev_signal = sig

            # 计算交易成本
            if i > 0:
                prev_pos = df.loc[i - 1, 'position']
                df.loc[i, 'cost'] = abs(current_pos - prev_pos) * cost_rate

        df.loc[i, 'position'] = current_pos

    # --- 成本（仅调仓日收取） --- #
    df['cost'] = abs(df['position'].diff().fillna(df['position'])) * cost_rate

    # 策略收益与超额收益
    df['strategy_ret'] = df['position'] * df[ret_col] - df['cost']
    df['excess_ret'] = df['strategy_ret'] - df[ret_col]

    # 绩效指标
    daily_ret = df['strategy_ret'].dropna()
    ann_ret = daily_ret.mean() * 252
    ann_vol = daily_ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

    ic_series, ir = calculate_ic_ir(
        df=df,
        factor_col=factor_col,
        ret_col='strategy_ret',
        window=20,
        plot=True
    )

    plot_monthly_position_distribution(df, signal_col='signal', date_col='date', title=f'{factor_col} monthly position distribution')

    regression_t_test(df, factor_name=f'{factor_col}', factor_col=factor_col, return_col='strategy_ret')

    results = {
        'nav_df': df[['date', 'strategy_ret', ret_col, 'excess_ret']],
        'perf': {
            'AnnRet': round(ann_ret, 4),
            'AnnVol': round(ann_vol, 4),
            'Sharpe': round(sharpe, 4),
        },
        'plot_path': f'{factor_col}_ts_backtest_daily_returns.png'
    }

    return results

def plot_strategy_vs_benchmark(df, factor_name, stage='train'):
    """
    对 DataFrame 的 'strategy_ret' 和 'returns' 绘制累计收益对比
    """
    df = df.copy()
    # 确保 'date' 列
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # 计算累积收益
    df['cum_strategy']  = df['strategy_ret'].cumsum()
    df['cum_benchmark'] = df['returns'].cumsum()
    df['cum_excess']    = df['cum_strategy'] - df['cum_benchmark']

    df['idx'] = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
    ax.plot(df['idx'], df['cum_strategy'],  label='Strategy', linewidth=1.2)
    ax.plot(df['idx'], df['cum_benchmark'], label='Benchmark', linewidth=1.2)
    ax.plot(df['idx'], df['cum_excess'],    label='Excess',    linewidth=1.2)

    ax.set_title(f'{factor_name} - {stage.capitalize()} Set: Strategy vs Benchmark')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.5)

    # 每 3 个月做一次刻度：约 63 个交易日
    step = 63
    xticks = df['idx'][::step]
    xticklabels = df['date'].dt.strftime('%Y-%m')[::step]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=45, ha='right')

    plt.tight_layout()
    path = f'{factor_name}_{stage}_strategy_vs_benchmark.png'
    plt.savefig(path)
    plt.show()
    plt.close()