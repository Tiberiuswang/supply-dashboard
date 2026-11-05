import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title='供应链履约看板（回放模拟）', layout='wide')
st.title('供应链履约看板')
st.caption('数据：DataCo 公开数据集（含模拟成分）。按订单时间回放历史数据，并非真实实时数据。')


@st.cache_data
def load():
    df = pd.read_csv('orders_clean.csv', parse_dates=['order_date'])
    return df.sort_values('order_date').reset_index(drop=True)


df = load()
modes = sorted(df['ship_mode'].unique())

# ---------- 侧边栏 ----------
st.sidebar.header('回放设置')
batch = st.sidebar.slider('每批订单行数', 500, 5000, 1000, 500)
wait = st.sidebar.slider('刷新间隔（秒）', 0.1, 3.0, 0.5, 0.1)
threshold = st.sidebar.slider('延误率预警线', 0.3, 0.9, 0.6, 0.05)
run = st.sidebar.button('开始回放')
st.sidebar.header('报童模型参数（假设值）')
cu = st.sidebar.slider('晚到每天损失 Cu', 1.0, 10.0, 3.0, 0.5)
co = st.sidebar.slider('过长承诺每天损失 Co', 0.5, 5.0, 1.0, 0.5)


# ---------- 工具函数 ----------
def best_promise(days, cu, co):
    ratio = cu / (cu + co)
    return int(np.percentile(days, ratio * 100, method='higher'))


def expected_cost(days, T, cu, co):
    late = np.maximum(days - T, 0)
    slack = np.maximum(T - days, 0)
    return (cu * late + co * slack).mean()


def show(fig):
    st.pyplot(fig)
    plt.close(fig)   # 及时关闭，避免回放时内存越积越多


def heatmap_fig(data):
    pv = data.pivot_table(index='ship_mode', columns='region', values='is_late', aggfunc='mean')
    fig, ax = plt.subplots(figsize=(7, 3.5))
    im = ax.imshow(pv.values, cmap='Reds', vmin=0, vmax=1, aspect='auto')   # 固定色阶，帧与帧可比
    ax.set_xticks(range(len(pv.columns)))
    ax.set_xticklabels(pv.columns, rotation=60, ha='right', fontsize=7)
    ax.set_yticks(range(len(pv.index)))
    ax.set_yticklabels(pv.index, fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def pareto_fig(data):
    late = data[data['is_late'] == 1]
    g = late.groupby(['ship_mode', 'region']).size().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    if g.empty:
        return fig
    cum = (g.cumsum() / g.sum()).head(10)
    top = g.head(10)
    labels = [a + ' / ' + b for a, b in top.index]
    ax.bar(range(len(top)), top.values, color='#c0392b')
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=6)
    ax2 = ax.twinx()
    ax2.plot(range(len(top)), cum.values, color='black', marker='o')
    ax2.set_ylim(0, 1.05)
    fig.tight_layout()
    return fig


tab1, tab2, tab3 = st.tabs(['实时回放', '整体分析', '承诺天数优化'])

# ---------- 页签2：整体分析（静态，先定义） ----------
with tab2:
    st.subheader('月度延误率与 3 个月移动平均')
    month = df['order_date'].dt.to_period('M').astype(str)
    m = df.groupby(month)['is_late'].mean()
    st.line_chart(pd.DataFrame({'月延误率': m, '3月移动平均': m.rolling(3).mean()}))

    left, right = st.columns(2)
    with left:
        st.subheader('延误天数分布（实际 - 承诺）')
        st.bar_chart(df['delay_days'].value_counts().sort_index())
    with right:
        st.subheader('平均承诺天数 vs 平均实际天数')
        st.bar_chart(df.groupby('ship_mode')[['days_sched', 'days_real']].mean(), stack=False)

    st.subheader('损失测算（赔付比例为假设值）')
    pay = st.slider('延误订单赔付/折扣比例', 0.0, 0.2, 0.05, 0.01)
    total_amt = df['amount'].sum()
    late_amt_all = df.loc[df['is_late'] == 1, 'amount'].sum()
    k1, k2 = st.columns(2)
    k1.metric('延误订单金额占比', f'{late_amt_all / total_amt:.1%}')
    k2.metric('估算赔付损失', f'{late_amt_all * pay:,.0f}')

# ---------- 页签3：报童模型（静态，先定义） ----------
with tab3:
    st.subheader('报童模型：按成本比例确定最优承诺天数（前70%训练，后30%验证）')
    cut = int(len(df) * 0.7)
    train, test = df.iloc[:cut], df.iloc[cut:]
    rows = []
    promise = {}
    for mode in modes:
        tr = train.loc[train['ship_mode'] == mode, 'days_real'].values
        te = test[test['ship_mode'] == mode]
        T = best_promise(tr, cu, co)
        promise[mode] = T
        rows.append({
            '配送方式': mode,
            '原平均承诺': te['days_sched'].mean(),
            '新承诺天数': T,
            '原延误率': (te['days_real'] > te['days_sched']).mean(),
            '新延误率': (te['days_real'] > T).mean(),
        })
    res = pd.DataFrame(rows).set_index('配送方式')

    new_T = test['ship_mode'].map(promise)
    old_all = (test['days_real'] > test['days_sched']).mean()
    new_all = (test['days_real'] > new_T).mean()
    extra = (new_T - test['days_sched']).mean()
    q1, q2, q3 = st.columns(3)
    q1.metric('验证期整体延误率（原承诺）', f'{old_all:.1%}')
    q2.metric('验证期整体延误率（新承诺）', f'{new_all:.1%}')
    q3.metric('平均承诺天数延长', f'{extra:.2f} 天')
    st.caption('注意：延误率下降是以承诺变长为代价，数据无法测算对转化率的影响。')

    st.dataframe(res.round(3))
    st.subheader('新旧承诺的延误率对比（验证期）')
    st.bar_chart(res[['原延误率', '新延误率']], stack=False)

    st.subheader('敏感性：Cu/Co 比值变化时的最优承诺天数')
    sens = {}
    for k in [1, 2, 3, 4, 5]:
        sens['Cu/Co=' + str(k)] = {
            mode: best_promise(train.loc[train['ship_mode'] == mode, 'days_real'].values, k, 1)
            for mode in modes
        }
    st.line_chart(pd.DataFrame(sens).T)

    st.subheader('期望成本曲线（最低点即最优承诺天数）')
    default_idx = modes.index('Standard Class') if 'Standard Class' in modes else 0
    sel = st.selectbox('选择配送方式', modes, index=default_idx)
    days = train.loc[train['ship_mode'] == sel, 'days_real'].values
    Ts = list(range(max(int(days.min()) - 1, 0), int(days.max()) + 3))
    cost = pd.Series([expected_cost(days, T, cu, co) for T in Ts], index=Ts)
    st.line_chart(cost)

# ---------- 页签1：实时回放（循环放最后） ----------
with tab1:
    box = st.empty()
    if not run:
        box.info('在左侧点击「开始回放」。')
    else:
        total = int(np.ceil(len(df) / batch))
        for step in range(1, total + 1):
            shown = df.iloc[: step * batch]
            rate = shown['is_late'].mean()
            late_amt = shown.loc[shown['is_late'] == 1, 'amount'].sum()
            last_day = shown['order_date'].iloc[-1]
            per_batch = shown['is_late'].groupby(np.arange(len(shown)) // batch).mean()
            by_mode = shown.groupby('ship_mode')['is_late'].mean()

            with box.container():
                c1, c2, c3, c4 = st.columns(4)
                c1.metric('回放进度', f'{step / total:.0%}')
                c2.metric('当前订单日期', f'{last_day:%Y-%m-%d}')
                c3.metric('累计延误率', f'{rate:.1%}')
                c4.metric('延误涉及金额', f'{late_amt:,.0f}')
                if rate > threshold:
                    st.error('延误率超过预警线')

                a, b = st.columns(2)
                with a:
                    st.subheader('逐批延误率（含预警线）')
                    st.line_chart(pd.DataFrame({'延误率': per_batch.values, '预警线': threshold}))
                with b:
                    st.subheader('各配送方式延误率')
                    st.bar_chart(by_mode)

                c, e = st.columns(2)
                with c:
                    st.subheader('配送方式 × 地区延误热力图')
                    show(heatmap_fig(shown))
                with e:
                    st.subheader('延误帕累托（前10个组合）')
                    show(pareto_fig(shown))
            time.sleep(wait)
        st.success('回放结束')
