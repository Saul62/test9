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

# 特征名称（只使用14个指定变量）
feature_names_display = [
    'age', 'LDL', 'bPRL', 'bE2', 'AMH', 'S_Dose', 'T_Dose',
    'D5_FSH', 'D5_LH', 'HCG_E2', 'HCG_LH', 'Ocytes', 'BFR', 'Cycles'
]

# 中文特征名称
feature_names_cn = [
    '女方年龄', '低密度脂蛋白胆固醇', '基线泌乳素', '基线雌二醇', '抗缪勒氏激素',
    '促性腺激素起始剂量', '促性腺激素总剂量', '促排第5天FSH', '促排第5天LH',
    'HCG日雌二醇', 'HCG日促黄体生成素', '获卵数', '囊胚形成率', '移植总周期数'
]

feature_dict = dict(zip(feature_names_display, feature_names_cn))

# 变量说明字典（只包含14个指定变量）
variable_descriptions = {
    'age': '女方年龄（岁）',
    'LDL': '低密度脂蛋白胆固醇（mmol/L）',
    'bPRL': '基线泌乳素（ng/mL）',
    'bE2': '基线雌二醇（pg/mL）',
    'AMH': '抗缪勒氏激素（ng/mL）',
    'S_Dose': '促性腺激素起始剂量（IU）',
    'T_Dose': '促性腺激素总剂量（IU）',
    'D5_FSH': '促排第5天促卵泡刺激素（mIU/mL）',
    'D5_LH': '促排第5天促黄体生成素（mIU/mL）',
    'HCG_E2': 'HCG日雌二醇（pg/mL）',
    'HCG_LH': 'HCG日促黄体生成素（mIU/mL）',
    'Ocytes': '获卵数（个）',
    'BFR': '囊胚形成率（%）',
    'Cycles': '移植总周期数（次）'
}

# 加载XGBoost模型和相关文件
@st.cache_resource
def load_model():
    # 加载XGBoost模型
    model = joblib.load('最佳模型/best_xgboost_model.pkl')

    # 加载标准化器
    scaler = joblib.load('数据/scaler.pkl')

    # 加载特征列名
    with open('最佳模型/feature_columns.pkl', 'rb') as f:
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
    st.markdown("### 请填写以下14个关键指标")

    # 创建标签页来组织输入
    tab1, tab2, tab3, tab4 = st.tabs(["病人基线信息", "促排过程监测", "触发排卵指标", "胚胎检测与移植"])

    with tab1:
        st.subheader("病人基线信息")
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("女方年龄（岁）", min_value=18, max_value=50, value=30)
            ldl = st.number_input("低密度脂蛋白胆固醇（mmol/L）", min_value=1.0, max_value=8.0, value=2.8, step=0.1)
            bprl = st.number_input("基线泌乳素（ng/mL）", min_value=1.0, max_value=100.0, value=15.0, step=0.1)

        with col2:
            be2 = st.number_input("基线雌二醇（pg/mL）", min_value=10.0, max_value=200.0, value=40.0, step=1.0)
            amh = st.number_input("抗缪勒氏激素（ng/mL）", min_value=0.1, max_value=20.0, value=3.0, step=0.1)

    with tab2:
        st.subheader("促排过程监测")
        col1, col2 = st.columns(2)

        with col1:
            s_dose = st.number_input("促性腺激素起始剂量（IU）", min_value=75, max_value=450, value=225)
            t_dose = st.number_input("促性腺激素总剂量（IU）", min_value=500, max_value=5000, value=2250)

        with col2:
            d5_fsh = st.number_input("促排第5天促卵泡刺激素（mIU/mL）", min_value=1.0, max_value=50.0, value=8.0, step=0.1)
            d5_lh = st.number_input("促排第5天促黄体生成素（mIU/mL）", min_value=0.5, max_value=30.0, value=3.0, step=0.1)
    
    with tab3:
        st.subheader("触发排卵指标")
        col1, col2 = st.columns(2)

        with col1:
            hcg_e2 = st.number_input("HCG日雌二醇（pg/mL）", min_value=500.0, max_value=8000.0, value=2000.0, step=50.0)

        with col2:
            hcg_lh = st.number_input("HCG日促黄体生成素（mIU/mL）", min_value=0.1, max_value=20.0, value=1.0, step=0.1)

    with tab4:
        st.subheader("胚胎检测指标与移植周期数")
        col1, col2 = st.columns(2)

        with col1:
            ocytes = st.number_input("获卵数（个）", min_value=1, max_value=50, value=12)
            bfr = st.number_input("囊胚形成率（%）", min_value=0.0, max_value=100.0, value=40.0, step=1.0)

        with col2:
            cycles = st.number_input("移植总周期数（次）", min_value=1, max_value=10, value=1)

    # 创建预测按钮
    predict_button = st.button("预测累积活产率", type="primary")

    if predict_button:
        # 收集14个输入特征
        features = [
            age, ldl, bprl, be2, amh, s_dose, t_dose,
            d5_fsh, d5_lh, hcg_e2, hcg_lh, ocytes, bfr, cycles
        ]

        # 转换为DataFrame（只包含14个特征列）
        input_df = pd.DataFrame([features], columns=feature_columns)

        # 标准化连续变量（所有14个变量都是连续变量）
        continuous_vars = ['age', 'LDL', 'bPRL', 'bE2', 'AMH', 'S_Dose', 'T_Dose',
                          'D5_FSH', 'D5_LH', 'HCG_E2', 'HCG_LH', 'Ocytes', 'BFR', 'Cycles']

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

            # 获取SHAP值（现在没有ID列了）
            for i, feature in enumerate(feature_names_display):
                # 在input_df中查找对应的特征
                feature_values.append(float(input_df[feature].iloc[0]))
                # SHAP值现在应该是一维数组，直接使用索引
                impact_value = float(shap_value[i])
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

                # 创建英文特征名（确保云端显示正常）
                english_feature_map = {
                    'Insemination': 'Treatment_Method', 'Complication': 'Complication',
                    'Years': 'Infertility_Years', 'Type': 'Infertility_Type',
                    'age': 'Female_Age', 'BMI': 'BMI', 'AMH': 'AMH', 'AFC': 'AFC',
                    'FBG': 'Fasting_Glucose', 'TC': 'Total_Cholesterol', 'TG': 'Triglycerides',
                    'HDL': 'HDL_Cholesterol', 'LDL': 'LDL_Cholesterol',
                    'bFSH': 'Baseline_FSH', 'bLH': 'Baseline_LH', 'bPRL': 'Baseline_PRL',
                    'bE2': 'Baseline_E2', 'bP': 'Baseline_P', 'bT': 'Baseline_T',
                    'D3_FSH': 'Day3_FSH', 'D3_LH': 'Day3_LH', 'D3_E2': 'Day3_E2',
                    'D5_FSH': 'Day5_FSH', 'D5_LH': 'Day5_LH', 'D5_E2': 'Day5_E2',
                    'COS': 'Stimulation_Protocol', 'S_Dose': 'Starting_Dose',
                    'T_Days': 'Treatment_Days', 'T_Dose': 'Total_Dose',
                    'HCG_LH': 'HCG_Day_LH', 'HCG_E2': 'HCG_Day_E2', 'HCG_P': 'HCG_Day_P',
                    'Ocytes': 'Retrieved_Oocytes', 'MII': 'MII_Rate', '2PN': 'Fertilization_Rate',
                    'CR': 'Cleavage_Rate', 'GVE': 'Good_Embryo_Rate',
                    'BFR': 'Blastocyst_Rate', 'Stage': 'Transfer_Stage',
                    'Cycles': 'Transfer_Cycles'
                }

                english_names = [english_feature_map.get(f, f) for f in feature_names_display]

                # 尝试使用中文特征名，如果失败则使用英文
                try:
                    # 首先尝试中文特征名
                    shap.waterfall_plot(
                        shap.Explanation(
                            values=shap_value,  # 现在没有ID列了
                            base_values=expected_value,
                            data=input_df.iloc[0].values,  # 现在没有ID列了
                            feature_names=[feature_dict.get(f, f) for f in feature_names_display]
                        ),
                        max_display=14,  # 显示所有14个特征
                        show=False
                    )
                    st.success("✅ 瀑布图使用中文特征名显示")
                except Exception as chinese_error:
                    st.warning("中文特征名显示失败，使用英文特征名")
                    # 如果中文失败，使用英文特征名
                    shap.waterfall_plot(
                        shap.Explanation(
                            values=shap_value,
                            base_values=expected_value,
                            data=input_df.iloc[0].values,
                            feature_names=english_names
                        ),
                        max_display=14,
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

                sorted_idx = np.argsort(np.abs(shap_value))[-14:]  # 显示所有14个特征

                bars = plt.barh(range(len(sorted_idx)), shap_value[sorted_idx])

                # 设置y轴标签（特征名）
                feature_labels = [feature_dict.get(feature_names_display[i], feature_names_display[i]) for i in sorted_idx]
                plt.yticks(range(len(sorted_idx)), feature_labels)

                plt.xlabel('SHAP值')
                plt.title('特征对累积活产预测的影响')

                # 为正负值设置不同颜色
                for i, bar in enumerate(bars):
                    if shap_value[sorted_idx[i]] >= 0:
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
                    shap_value,  # 现在没有ID列了
                    input_df.iloc[0],  # 现在没有ID列了
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
                # 现在没有ID列了
                importance_df = pd.DataFrame({
                    '特征': [feature_dict.get(f, f) for f in feature_names_display],
                    '重要性': feature_importance  # 现在没有ID列了
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
