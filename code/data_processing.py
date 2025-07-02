# data_processing.py
import os
import pandas as pd

def load_and_extract_features(data_dir='../oulad', module='FFF', presentation='2013J'):
    # 加载数据
    student_info = pd.read_csv(os.path.join(data_dir, 'studentInfo.csv'))
    student_vle = pd.read_csv(os.path.join(data_dir, 'studentVle.csv'))
    student_assess = pd.read_csv(os.path.join(data_dir, 'studentAssessment.csv'))
    vle = pd.read_csv(os.path.join(data_dir, 'vle.csv'))
    student_reg = pd.read_csv(os.path.join(data_dir, 'studentRegistration.csv'))
    courses = pd.read_csv(os.path.join(data_dir, 'courses.csv'))

    # 筛选指定课程
    student_info = student_info[
        (student_info['code_module'] == module) & 
        (student_info['code_presentation'] == presentation)
    ]
    student_ids = student_info['id_student'].unique()
    student_vle = student_vle[student_vle['id_student'].isin(student_ids)]
    student_assess = student_assess[student_assess['id_student'].isin(student_ids)]
    student_reg = student_reg[student_reg['id_student'].isin(student_ids)]

    # 前4周行为特征
    vle_4weeks = student_vle[student_vle['date'] <= 28]
    clicks_sum = vle_4weeks.groupby('id_student')['sum_click'].sum().reset_index()
    clicks_sum.columns = ['id_student', 'total_clicks_4w']
    active_days = vle_4weeks.groupby('id_student')['date'].nunique().reset_index()
    active_days.columns = ['id_student', 'active_days_4w']
    click_density = clicks_sum.merge(active_days, on='id_student')
    click_density['click_density_4w'] = click_density['total_clicks_4w'] / (click_density['active_days_4w'] + 1e-5)

    # 资源类型点击特征
    vle_merged = vle_4weeks.merge(vle[['id_site', 'activity_type']], on='id_site', how='left')
    resource_type_counts = vle_merged.groupby(['id_student', 'activity_type'])['sum_click'].sum().unstack().fillna(0)
    resource_type_counts.columns = [f'resource_clicks_{col}_4w' for col in resource_type_counts.columns]
    resource_type_counts.reset_index(inplace=True)

    # 测验特征
    assess_4weeks = student_assess[student_assess['date_submitted'] <= 28]
    avg_score = assess_4weeks.groupby('id_student')['score'].agg(['mean', 'std', 'count']).reset_index()
    avg_score.columns = ['id_student', 'avg_score_4w', 'score_std_4w', 'quiz_count_4w']
    avg_score.fillna(0, inplace=True)

    # 注册信息特征
    student_reg['registration_days'] = student_reg['date_unregistration'].fillna(1000) - student_reg['date_registration']
    registration = student_reg[['id_student', 'date_registration', 'registration_days']]

    # 标签
    student_info['label'] = student_info['final_result'].apply(lambda x: 1 if x == 'Fail' else 0)
    label_df = student_info[['id_student', 'label']]

    # 合并全部特征
    dfs = [clicks_sum, active_days, click_density[['id_student', 'click_density_4w']],
           resource_type_counts, avg_score, registration, label_df]

    data = dfs[0]
    for df in dfs[1:]:
        data = data.merge(df, on='id_student', how='outer')

    data.fillna(0, inplace=True)

    # 特征列表
    feature_cols = [
        'total_clicks_4w', 'active_days_4w', 'click_density_4w',
        'avg_score_4w', 'score_std_4w', 'quiz_count_4w',
        'date_registration', 'registration_days'
    ] + [col for col in data.columns if col.startswith('resource_clicks_')]

    # 过滤无用特征
    low_importance_features = [
        'resource_clicks_sharedsubpage_4w', 'resource_clicks_dataplus_4w',
        'resource_clicks_questionnaire_4w', 'resource_clicks_htmlactivity_4w',
        'resource_clicks_dualpane_4w',
    ]
    feature_cols = [col for col in feature_cols if col not in low_importance_features]

    return data[feature_cols], data['label']
