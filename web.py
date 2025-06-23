import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import matplotlib
import shap
import warnings

# 忽略不必要的警告
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# 修复NumPy bool弃用问题
if not hasattr(np, 'bool'):
    np.bool = bool

# 全局设置matplotlib字体，确保负号正常显示
def setup_chinese_font():
    """设置中文字体"""
    try:
        import matplotlib.font_manager as fm

        # 尝试多种中文字体
        chinese_fonts = [
            'WenQuanYi Zen Hei',  # 文泉驿正黑（Linux常用）
            'WenQuanYi Micro Hei',  # 文泉驿微米黑
            'SimHei',  # 黑体
            'Microsoft YaHei',  # 微软雅黑
            'PingFang SC',  # 苹果字体
            'Hiragino Sans GB',  # 冬青黑体
            'Noto Sans CJK SC',  # Google Noto字体
            'Source Han Sans SC'  # 思源黑体
        ]

        # 获取系统可用字体
        available_fonts = [f.name for f in fm.fontManager.ttflist]

        # 查找可用的中文字体
        for font in chinese_fonts:
            if font in available_fonts:
                matplotlib.rcParams['font.sans-serif'] = [font, 'DejaVu Sans', 'Arial']
                matplotlib.rcParams['font.family'] = 'sans-serif'
                print(f"使用中文字体: {font}")
                return font

        # 如果没有找到中文字体，使用默认字体
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
        matplotlib.rcParams['font.family'] = 'sans-serif'
        print("未找到中文字体，使用默认字体")
        return None

    except Exception as e:
        print(f"字体设置失败: {e}")
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        matplotlib.rcParams['font.family'] = 'sans-serif'
        return None

# 设置字体和负号显示
chinese_font = setup_chinese_font()
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置页面标题和布局
st.set_page_config(
    page_title="多囊卵巢综合征患者辅助生殖累积活产率预测系统",
    page_icon="🏥",
    layout="wide"
)

# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义全局变量
global feature_names, feature_dict, variable_descriptions

# 特征名称（用于显示，不包含ID列）
feature_names_display = [
    'Insemination', 'Complication', 'Years', 'Type', 'age', 'BMI', 'AMH', 'AFC', 'FBG',
    'TC', 'TG', 'HDL', 'LDL', 'bFSH', 'bLH', 'bPRL', 'bE2', 'bP', 'bT',
    'D3_FSH', 'D3_LH', 'D3_E2', 'D5_FSH', 'D5_LH', 'D5_E2', 'COS', 'S_Dose',
    'T_Days', 'T_Dose', 'HCG_LH', 'HCG_E2', 'HCG_P', 'Ocytes', 'MII', '2PN',
    'CR', 'GVE', 'BFR', 'Stage', 'Cycles'
]

# 中文特征名称
feature_names_cn = [
    '治疗方案', '合并诊断', '不孕年限', '不孕类型', '女方年龄', '身高体重指数', '抗苗勒管激素', '窦卵泡数', '空腹血糖',
    '总胆固醇', '甘油三酯', '高密度脂蛋白胆固醇', '低密度脂蛋白胆固醇', '基础卵泡刺激素', '基础黄体生成素', '基础泌乳素', '基础雌二醇', '基础孕激素', '基础雄激素',
    '促排第3天促卵泡刺激素', '促排第3天促黄体生成素', '促排第3天雌二醇', '促排第5天促卵泡刺激素', '促排第5天促黄体生成素', '促排第5天雌二醇', '促排方案', 'Gn起始剂量',
    '促排卵天数', 'Gn总剂量', 'HCG日促黄体生成素', 'HCG日雌二醇', 'HCG日孕激素', '获卵数', 'MII率', '2PN率',
    '卵裂率', '优质胚胎率', '囊胚形成率', '移植期别', '移植总周期数'
]

feature_dict = dict(zip(feature_names_display, feature_names_cn))

# 变量说明字典
variable_descriptions = {
    'Insemination': '1=常规IVF，2=ICSI',
    'Complication': '1=无，2=合并女方因素，3=合并男方因素，4=合并女方+男方因素',
    'Years': '不孕年限（年）',
    'Type': '1=原发不孕，2=继发不孕',
    'age': '女方年龄（岁）',
    'BMI': '身高体重指数（kg/m²）',
    'AMH': '抗苗勒管激素（ng/mL）',
    'AFC': '窦卵泡数（个）',
    'FBG': '空腹血糖（mmol/L）',
    'TC': '总胆固醇（mmol/L）',
    'TG': '甘油三酯（mmol/L）',
    'HDL': '高密度脂蛋白胆固醇（mmol/L）',
    'LDL': '低密度脂蛋白胆固醇（mmol/L）',
    'bFSH': '基础卵泡刺激素（mIU/mL）',
    'bLH': '基础黄体生成素（mIU/mL）',
    'bPRL': '基础泌乳素（ng/mL）',
    'bE2': '基础雌二醇（pg/mL）',
    'bP': '基础孕激素（ng/mL）',
    'bT': '基础雄激素（ng/mL）',
    'D3_FSH': '促排第3天促卵泡刺激素（mIU/mL）',
    'D3_LH': '促排第3天促黄体生成素（mIU/mL）',
    'D3_E2': '促排第3天雌二醇（pg/mL）',
    'D5_FSH': '促排第5天促卵泡刺激素（mIU/mL）',
    'D5_LH': '促排第5天促黄体生成素（mIU/mL）',
    'D5_E2': '促排第5天雌二醇（pg/mL）',
    'COS': '1=拮抗剂方案，2=黄体期长方案，3=其他',
    'S_Dose': 'Gn起始剂量（IU）',
    'T_Days': '促排卵天数（天）',
    'T_Dose': 'Gn总剂量（IU）',
    'HCG_LH': 'HCG日促黄体生成素（mIU/mL）',
    'HCG_E2': 'HCG日雌二醇（pg/mL）',
    'HCG_P': 'HCG日孕激素（ng/mL）',
    'Ocytes': '获卵数（个）',
    'MII': 'MII率（%）',
    '2PN': '2PN率（%）',
    'CR': '卵裂率（%）',
    'GVE': '优质胚胎率（%）',
    'BFR': '囊胚形成率（%）',
    'Stage': '1=新鲜周期移植，2=冷冻周期移植',
    'Cycles': '移植总周期数（次）'
}

# 加载XGBoost模型和相关文件
@st.cache_resource
def load_model():
    # 加载XGBoost模型
    model = joblib.load('best_xgboost_model.pkl')

    # 加载标准化器
    scaler = joblib.load('scaler.pkl')

    # 加载特征列名
    with open('feature_columns.pkl', 'rb') as f:
        feature_columns = pickle.load(f)

    return model, scaler, feature_columns

# 主应用
def main():
    global feature_names, feature_dict, variable_descriptions

    # 侧边栏标题
    st.sidebar.title("多囊卵巢综合征患者辅助生殖累积活产率预测系统V1.0")
    st.sidebar.image("https://img.freepik.com/free-vector/hospital-logo-design-vector-medical-cross_53876-136743.jpg", width=200)

    # 添加系统说明到侧边栏
    st.sidebar.markdown("""
    # 系统说明

    ## 关于本系统
    这是一个基于XGBoost算法的多囊卵巢综合征患者辅助生殖累积活产率预测系统，通过分析患者的临床指标和治疗过程数据来预测累积活产的可能性。

    ## 预测结果
    系统预测：
    - 累积活产概率
    - 无累积活产概率
    - 风险评估（低风险、中风险、高风险）

    ## 使用方法
    1. 在主界面填写患者的临床指标
    2. 点击预测按钮生成预测结果
    3. 查看预测结果和特征重要性分析

    ## 重要提示
    - 请确保患者信息输入准确
    - 所有字段都需要填写
    - 数值字段需要输入数字
    - 选择字段需要从选项中选择
    """)
    
    # 添加变量说明到侧边栏
    with st.sidebar.expander("变量说明"):
        for feature in feature_names_display:
            st.markdown(f"**{feature_dict[feature]}**: {variable_descriptions[feature]}")

    # 主页面标题
    st.title("多囊卵巢综合征患者辅助生殖累积活产率预测系统V1.0")
    st.markdown("### 基于XGBoost算法的累积活产率评估")

    # 加载模型
    try:
        model, scaler, feature_columns = load_model()
        st.sidebar.success("XGBoost模型加载成功！")
    except Exception as e:
        st.sidebar.error(f"模型加载失败: {e}")
        return
    
    # 创建输入表单
    st.header("患者信息输入")

    # 创建标签页来组织输入
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["基本信息", "基础激素", "促排监测", "胚胎实验室", "移植策略"])

    with tab1:
        st.subheader("基本信息")
        col1, col2 = st.columns(2)

        with col1:
            insemination = st.selectbox("治疗方案", options=[1, 2], format_func=lambda x: "常规IVF" if x == 1 else "ICSI")
            complication = st.selectbox("合并诊断", options=[1, 2, 3, 4],
                                      format_func=lambda x: "无" if x == 1 else "合并女方因素" if x == 2 else "合并男方因素" if x == 3 else "合并女方+男方因素")
            years = st.number_input("不孕年限（年）", min_value=0.5, max_value=20.0, value=3.0, step=0.5)
            type_infertility = st.selectbox("不孕类型", options=[1, 2], format_func=lambda x: "原发不孕" if x == 1 else "继发不孕")
            age = st.number_input("女方年龄（岁）", min_value=18, max_value=50, value=30)

        with col2:
            bmi = st.number_input("身高体重指数（kg/m²）", min_value=15.0, max_value=40.0, value=23.0, step=0.1)
            amh = st.number_input("抗苗勒管激素（ng/mL）", min_value=0.1, max_value=20.0, value=3.0, step=0.1)
            afc = st.number_input("窦卵泡数（个）", min_value=1, max_value=50, value=12)
            fbg = st.number_input("空腹血糖（mmol/L）", min_value=3.0, max_value=15.0, value=5.0, step=0.1)

    with tab2:
        st.subheader("血脂和基础激素")
        col1, col2 = st.columns(2)

        with col1:
            tc = st.number_input("总胆固醇（mmol/L）", min_value=2.0, max_value=10.0, value=4.5, step=0.1)
            tg = st.number_input("甘油三酯（mmol/L）", min_value=0.5, max_value=10.0, value=1.5, step=0.1)
            hdl = st.number_input("高密度脂蛋白胆固醇（mmol/L）", min_value=0.5, max_value=3.0, value=1.3, step=0.1)
            ldl = st.number_input("低密度脂蛋白胆固醇（mmol/L）", min_value=1.0, max_value=8.0, value=2.8, step=0.1)
            bfsh = st.number_input("基础卵泡刺激素（mIU/mL）", min_value=1.0, max_value=50.0, value=6.0, step=0.1)

        with col2:
            blh = st.number_input("基础黄体生成素（mIU/mL）", min_value=0.5, max_value=30.0, value=5.0, step=0.1)
            bprl = st.number_input("基础泌乳素（ng/mL）", min_value=1.0, max_value=100.0, value=15.0, step=0.1)
            be2 = st.number_input("基础雌二醇（pg/mL）", min_value=10.0, max_value=200.0, value=40.0, step=1.0)
            bp = st.number_input("基础孕激素（ng/mL）", min_value=0.1, max_value=10.0, value=0.8, step=0.1)
            bt = st.number_input("基础雄激素（ng/mL）", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
    
    with tab3:
        st.subheader("促排过程监测")
        col1, col2 = st.columns(2)

        with col1:
            d3_fsh = st.number_input("促排第3天促卵泡刺激素（mIU/mL）", min_value=1.0, max_value=50.0, value=6.0, step=0.1)
            d3_lh = st.number_input("促排第3天促黄体生成素（mIU/mL）", min_value=0.5, max_value=30.0, value=5.0, step=0.1)
            d3_e2 = st.number_input("促排第3天雌二醇（pg/mL）", min_value=10.0, max_value=500.0, value=50.0, step=1.0)
            d5_fsh = st.number_input("促排第5天促卵泡刺激素（mIU/mL）", min_value=1.0, max_value=50.0, value=8.0, step=0.1)
            d5_lh = st.number_input("促排第5天促黄体生成素（mIU/mL）", min_value=0.5, max_value=30.0, value=3.0, step=0.1)

        with col2:
            d5_e2 = st.number_input("促排第5天雌二醇（pg/mL）", min_value=50.0, max_value=2000.0, value=200.0, step=10.0)
            cos = st.selectbox("促排方案", options=[1, 2, 3],
                             format_func=lambda x: "拮抗剂方案" if x == 1 else "黄体期长方案" if x == 2 else "其他")
            s_dose = st.number_input("Gn起始剂量（IU）", min_value=75, max_value=450, value=225)
            t_days = st.number_input("促排卵天数（天）", min_value=5, max_value=20, value=10)
            t_dose = st.number_input("Gn总剂量（IU）", min_value=500, max_value=5000, value=2250)

    with tab4:
        st.subheader("HCG日指标和胚胎实验室")
        col1, col2 = st.columns(2)

        with col1:
            hcg_lh = st.number_input("HCG日促黄体生成素（mIU/mL）", min_value=0.1, max_value=20.0, value=1.0, step=0.1)
            hcg_e2 = st.number_input("HCG日雌二醇（pg/mL）", min_value=500.0, max_value=8000.0, value=2000.0, step=50.0)
            hcg_p = st.number_input("HCG日孕激素（ng/mL）", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
            ocytes = st.number_input("获卵数（个）", min_value=1, max_value=50, value=12)
            mii = st.number_input("MII率（%）", min_value=0.0, max_value=100.0, value=80.0, step=1.0)

        with col2:
            pn2 = st.number_input("2PN率（%）", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
            cr = st.number_input("卵裂率（%）", min_value=0.0, max_value=100.0, value=90.0, step=1.0)
            gve = st.number_input("优质胚胎率（%）", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
            bfr = st.number_input("囊胚形成率（%）", min_value=0.0, max_value=100.0, value=40.0, step=1.0)

    with tab5:
        st.subheader("移植策略")
        col1, col2 = st.columns(2)

        with col1:
            stage = st.selectbox("移植期别", options=[1, 2], format_func=lambda x: "新鲜周期移植" if x == 1 else "冷冻周期移植")

        with col2:
            cycles = st.number_input("移植总周期数（次）", min_value=1, max_value=10, value=1)

    # 创建预测按钮
    predict_button = st.button("预测累积活产率", type="primary")

    if predict_button:
        # 收集所有输入特征 - 按照训练数据的顺序（包含ID列）
        features = [
            1,  # ID列（可以是任意值）
            insemination, complication, years, type_infertility, age, bmi, amh, afc, fbg,
            tc, tg, hdl, ldl, bfsh, blh, bprl, be2, bp, bt,
            d3_fsh, d3_lh, d3_e2, d5_fsh, d5_lh, d5_e2, cos, s_dose,
            t_days, t_dose, hcg_lh, hcg_e2, hcg_p, ocytes, mii, pn2,
            cr, gve, bfr, stage, cycles
        ]

        # 转换为DataFrame（包含所有特征列）
        input_df = pd.DataFrame([features], columns=feature_columns)

        # 标准化连续变量
        continuous_vars = ['Years', 'age', 'BMI', 'AMH', 'AFC', 'FBG', 'TC', 'TG', 'HDL', 'LDL',
                          'bFSH', 'bLH', 'bPRL', 'bE2', 'bP', 'bT', 'D3_FSH', 'D3_LH', 'D3_E2',
                          'D5_FSH', 'D5_LH', 'D5_E2', 'S_Dose', 'T_Days', 'T_Dose', 'HCG_LH',
                          'HCG_E2', 'HCG_P', 'Ocytes', 'MII', '2PN', 'CR', 'GVE', 'BFR']

        # 创建输入数据的副本用于标准化
        input_scaled = input_df.copy()
        input_scaled[continuous_vars] = scaler.transform(input_df[continuous_vars])

        # 进行预测
        prediction = model.predict_proba(input_scaled)[0]
        no_birth_prob = prediction[0]
        birth_prob = prediction[1]
        
        # 显示预测结果
        st.header("累积活产率预测结果")

        # 使用进度条显示概率
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("无累积活产概率")
            st.progress(float(no_birth_prob))
            st.write(f"{no_birth_prob:.2%}")

        with col2:
            st.subheader("累积活产概率")
            st.progress(float(birth_prob))
            st.write(f"{birth_prob:.2%}")

        # 风险评估
        risk_level = "低概率" if birth_prob < 0.3 else "中等概率" if birth_prob < 0.7 else "高概率"
        risk_color = "red" if birth_prob < 0.3 else "orange" if birth_prob < 0.7 else "green"

        st.markdown(f"### 累积活产评估: <span style='color:{risk_color}'>{risk_level}</span>", unsafe_allow_html=True)
        

        
        # 添加模型解释
        st.write("---")
        st.subheader("模型解释")

        try:
            # 创建SHAP解释器
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)

            # 处理SHAP值格式 - 形状为(1, 10, 2)表示1个样本，10个特征，2个类别
            if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
                # 取第一个样本的正类（DKD类，索引1）的SHAP值
                shap_value = shap_values[0, :, 1]  # 形状变为(10,)
                expected_value = explainer.expected_value[1]  # 正类的期望值
            elif isinstance(shap_values, list):
                # 如果是列表格式，取正类的SHAP值
                shap_value = np.array(shap_values[1][0])
                expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
            else:
                shap_value = np.array(shap_values[0])
                expected_value = explainer.expected_value

            # 特征贡献分析表格
            st.subheader("特征贡献分析")

            # 创建贡献表格
            feature_values = []
            feature_impacts = []

            # 获取SHAP值（跳过ID列）
            for i, feature in enumerate(feature_names_display):
                # 在input_df中查找对应的特征（跳过ID列）
                feature_values.append(float(input_df[feature].iloc[0]))
                # SHAP值现在应该是一维数组，需要跳过ID列对应的索引
                impact_value = float(shap_value[i+1])  # +1是因为跳过ID列
                feature_impacts.append(impact_value)

            shap_df = pd.DataFrame({
                '特征': [feature_dict.get(f, f) for f in feature_names_display],
                '数值': feature_values,
                '影响': feature_impacts
            })

            # 按绝对影响排序
            shap_df['绝对影响'] = shap_df['影响'].abs()
            shap_df = shap_df.sort_values('绝对影响', ascending=False)

            # 显示表格
            st.table(shap_df[['特征', '数值', '影响']])
            
            # SHAP瀑布图
            st.subheader("SHAP瀑布图")

            try:
                # 创建SHAP瀑布图
                import matplotlib.font_manager as fm

                # 尝试设置中文字体
                try:
                    # 尝试使用系统中文字体（包含Linux云端服务器常用字体）
                    chinese_fonts = [
                        'WenQuanYi Zen Hei',  # 文泉驿正黑（Linux常用）
                        'WenQuanYi Micro Hei',  # 文泉驿微米黑（Linux常用）
                        'Noto Sans CJK SC',  # Google Noto字体
                        'Source Han Sans SC',  # 思源黑体
                        'SimHei',  # 黑体
                        'Microsoft YaHei',  # 微软雅黑
                        'PingFang SC',  # 苹果字体
                        'Hiragino Sans GB'  # 冬青黑体
                    ]
                    available_fonts = [f.name for f in fm.fontManager.ttflist]

                    chinese_font = None
                    for font in chinese_fonts:
                        if font in available_fonts:
                            chinese_font = font
                            break

                    if chinese_font:
                        plt.rcParams['font.sans-serif'] = [chinese_font, 'DejaVu Sans']
                        plt.rcParams['font.family'] = 'sans-serif'
                    else:
                        # 如果没有中文字体，使用英文字体
                        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
                        plt.rcParams['font.family'] = 'sans-serif'

                except Exception:
                    # 字体设置失败，使用默认字体
                    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
                    plt.rcParams['font.family'] = 'sans-serif'

                plt.rcParams['axes.unicode_minus'] = False

                fig_waterfall = plt.figure(figsize=(12, 8))

                # 使用新版本的waterfall plot（跳过ID列）
                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_value[1:],  # 跳过ID列的SHAP值
                        base_values=expected_value,
                        data=input_df.iloc[0, 1:].values,  # 跳过ID列的数据
                        feature_names=[feature_dict.get(f, f) for f in feature_names_display]
                    ),
                    max_display=10,
                    show=False
                )

                # 手动设置中文字体和修复负号显示
                for ax in fig_waterfall.get_axes():
                    # 设置坐标轴标签字体
                    ax.tick_params(labelsize=10)

                    # 修复所有文本的字体和负号
                    for text in ax.texts:
                        text_content = text.get_text()
                        # 替换unicode minus
                        if '−' in text_content:
                            text.set_text(text_content.replace('−', '-'))
                        # 设置字体
                        if chinese_font:
                            text.set_fontfamily(chinese_font)
                        text.set_fontsize(10)

                    # 设置y轴标签字体
                    for label in ax.get_yticklabels():
                        if chinese_font:
                            label.set_fontfamily(chinese_font)
                        label.set_fontsize(10)

                    # 设置x轴标签字体
                    for label in ax.get_xticklabels():
                        if chinese_font:
                            label.set_fontfamily(chinese_font)
                        label.set_fontsize(10)

                plt.tight_layout()
                st.pyplot(fig_waterfall)
                plt.close(fig_waterfall)
            except Exception as e:
                st.error(f"无法生成瀑布图: {str(e)}")
                # 使用条形图作为替代（跳过ID列）
                fig_bar = plt.figure(figsize=(10, 6))

                # 设置中文字体
                try:
                    import matplotlib.font_manager as fm
                    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Hiragino Sans GB', 'WenQuanYi Micro Hei']
                    available_fonts = [f.name for f in fm.fontManager.ttflist]

                    chinese_font = None
                    for font in chinese_fonts:
                        if font in available_fonts:
                            chinese_font = font
                            break

                    if chinese_font:
                        plt.rcParams['font.sans-serif'] = [chinese_font, 'DejaVu Sans']
                    else:
                        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
                except Exception:
                    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']

                plt.rcParams['axes.unicode_minus'] = False

                shap_value_no_id = shap_value[1:]  # 跳过ID列
                sorted_idx = np.argsort(np.abs(shap_value_no_id))[-10:]

                bars = plt.barh(range(len(sorted_idx)), shap_value_no_id[sorted_idx])

                # 设置y轴标签（特征名）
                feature_labels = [feature_dict.get(feature_names_display[i], feature_names_display[i]) for i in sorted_idx]
                plt.yticks(range(len(sorted_idx)), feature_labels)

                plt.xlabel('SHAP值')
                plt.title('特征对累积活产预测的影响')

                # 为正负值设置不同颜色
                for i, bar in enumerate(bars):
                    if shap_value_no_id[sorted_idx[i]] >= 0:
                        bar.set_color('lightcoral')
                    else:
                        bar.set_color('lightblue')

                plt.tight_layout()
                st.pyplot(fig_bar)
                plt.close(fig_bar)

            # SHAP力图
            st.subheader("SHAP力图")

            try:
                # 使用官方SHAP力图，HTML格式
                import streamlit.components.v1 as components
                import matplotlib

                # 设置字体确保负号显示
                matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
                matplotlib.rcParams['axes.unicode_minus'] = False

                force_plot = shap.force_plot(
                    expected_value,
                    shap_value[1:],  # 跳过ID列
                    input_df.iloc[0, 1:],  # 跳过ID列
                    feature_names=[feature_dict.get(f, f) for f in feature_names_display]
                )

                # 获取SHAP的HTML内容，添加CSS来修复遮挡问题
                shap_html = f"""
                <head>
                    {shap.getjs()}
                    <style>
                        body {{
                            margin: 0;
                            padding: 20px 10px 40px 10px;
                            overflow: visible;
                        }}
                        .force-plot {{
                            margin: 20px 0 40px 0 !important;
                            padding: 20px 0 40px 0 !important;
                        }}
                        svg {{
                            margin: 20px 0 40px 0 !important;
                        }}
                        .tick text {{
                            margin-bottom: 20px !important;
                        }}
                        .force-plot-container {{
                            min-height: 200px !important;
                            padding-bottom: 50px !important;
                        }}
                    </style>
                </head>
                <body>
                    <div class="force-plot-container">
                        {force_plot.html()}
                    </div>
                </body>
                """

                # 增加更多高度空间
                components.html(shap_html, height=400, scrolling=False)

            except Exception as e:
                st.error(f"无法生成HTML力图: {str(e)}")
                st.info("请检查SHAP版本是否兼容")
            
        except Exception as e:
            st.error(f"无法生成SHAP解释: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            st.info("使用模型特征重要性作为替代")

            # 显示模型特征重要性
            st.write("---")
            st.subheader("特征重要性")

            # 从XGBoost模型获取特征重要性
            try:
                feature_importance = model.feature_importances_
                # 跳过ID列的重要性
                importance_df = pd.DataFrame({
                    '特征': [feature_dict.get(f, f) for f in feature_names_display],
                    '重要性': feature_importance[1:]  # 跳过ID列
                }).sort_values('重要性', ascending=False)

                fig, ax = plt.subplots(figsize=(12, 8))

                # 设置中文字体
                try:
                    import matplotlib.font_manager as fm
                    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Hiragino Sans GB', 'WenQuanYi Micro Hei']
                    available_fonts = [f.name for f in fm.fontManager.ttflist]

                    chinese_font = None
                    for font in chinese_fonts:
                        if font in available_fonts:
                            chinese_font = font
                            break

                    if chinese_font:
                        plt.rcParams['font.sans-serif'] = [chinese_font, 'DejaVu Sans']
                    else:
                        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
                except Exception:
                    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']

                plt.rcParams['axes.unicode_minus'] = False

                bars = plt.barh(range(len(importance_df)), importance_df['重要性'], color='skyblue')
                plt.yticks(range(len(importance_df)), importance_df['特征'])
                plt.xlabel('重要性')
                plt.ylabel('特征')
                plt.title('特征重要性')

                # 设置字体
                if 'chinese_font' in locals() and chinese_font:
                    ax.set_xlabel('重要性', fontfamily=chinese_font)
                    ax.set_ylabel('特征', fontfamily=chinese_font)
                    ax.set_title('特征重要性', fontfamily=chinese_font)

                    # 设置刻度标签字体
                    for label in ax.get_yticklabels():
                        label.set_fontfamily(chinese_font)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e2:
                st.error(f"无法显示特征重要性: {str(e2)}")

if __name__ == "__main__":
    main()
