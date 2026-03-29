"""통계 분석 스크립트 — UniversalBank (타겟: PLoan)"""
import json
import pandas as pd
import numpy as np
from scipy import stats
import scikit_posthocs as sp

# ── 데이터 로드 ──
DATA_PATH = 'cleaned/UniversalBank_cleaned.csv'
df = pd.read_csv(DATA_PATH)
print(f"데이터 로드: {df.shape[0]:,}행 × {df.shape[1]}열")


# ── 분석 함수 정의 ──

def analyze_two_group(df, x_var, y_var, val1, val2):
    """범주형(2그룹) — 연속형 차이 검정"""
    group1 = df[df[x_var] == val1][y_var].dropna()
    group2 = df[df[x_var] == val2][y_var].dropna()
    n = len(group1) + len(group2)

    # 정규성 검정 (n ≤ 5000 → Shapiro-Wilk)
    stat_n, p_n = stats.shapiro(pd.concat([group1, group2]))
    norm_test = 'Shapiro-Wilk'
    is_normal = p_n > 0.05

    # 등분산 검정 (Levene)
    stat_lev, p_lev = stats.levene(group1, group2)
    is_equal_var = p_lev > 0.05

    # 본 검정 자동 분기
    if is_normal and is_equal_var:
        stat, p = stats.ttest_ind(group1, group2, equal_var=True)
        test_name = '독립표본 t-test'
    elif is_normal:
        stat, p = stats.ttest_ind(group1, group2, equal_var=False)
        test_name = "Welch's t-test"
    else:
        stat, p = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        test_name = 'Mann-Whitney U'

    # 효과 크기 (Cohen's d)
    pooled_std = np.sqrt((group1.std()**2 + group2.std()**2) / 2)
    cohens_d = abs(group1.mean() - group2.mean()) / pooled_std if pooled_std > 0 else 0

    return {
        'title': f'{x_var}에 따른 {y_var} 차이 검정',
        'question': f'{x_var}에 따라 {y_var}에 유의한 차이가 있는가?',
        'x_var': x_var, 'y_var': y_var,
        'h0': f'{x_var}에 따른 {y_var}의 평균 차이가 없다',
        'h1': f'{x_var}에 따른 {y_var}의 평균 차이가 있다',
        'assumptions': {
            'normality': {'test': norm_test, 'statistic': float(stat_n), 'p_value': float(p_n), 'pass': bool(is_normal)},
            'equal_variance': {'test': 'Levene', 'statistic': float(stat_lev), 'p_value': float(p_lev), 'pass': bool(is_equal_var)},
        },
        'test_name': test_name,
        'statistic': float(stat), 'p_value': float(p),
        'conclusion': 'H0 기각' if p < 0.05 else 'H0 기각할 수 없음',
        'interpretation': f'두 그룹 간 유의한 차이가 {"있다" if p < 0.05 else "있다고 볼 수 없다"} (α=0.05)',
        'effect_size': {'name': "Cohen's d", 'value': float(cohens_d)},
        'posthoc': None,
        'group_stats': {
            str(val1): {'n': int(len(group1)), 'mean': float(group1.mean()), 'std': float(group1.std())},
            str(val2): {'n': int(len(group2)), 'mean': float(group2.mean()), 'std': float(group2.std())},
        },
    }


def analyze_multi_group(df, x_var, y_var):
    """범주형(3+그룹) — 연속형 차이 검정"""
    groups = {name: g[y_var].dropna().values for name, g in df.groupby(x_var)}
    group_list = list(groups.values())
    all_data = df[y_var].dropna()
    n = len(all_data)

    # 정규성 검정
    stat_n, p_n = stats.shapiro(all_data)
    norm_test = 'Shapiro-Wilk'
    is_normal = p_n > 0.05

    # 등분산 검정
    stat_lev, p_lev = stats.levene(*group_list)
    is_equal_var = p_lev > 0.05

    # 본 검정 자동 분기
    if is_normal and is_equal_var:
        stat, p = stats.f_oneway(*group_list)
        test_name = 'One-way ANOVA'
    elif is_normal:
        res = stats.alexandergovern(*group_list)
        stat, p = res.statistic, res.pvalue
        test_name = "Welch's ANOVA (Alexander-Govern)"
    else:
        stat, p = stats.kruskal(*group_list)
        test_name = 'Kruskal-Wallis'

    # 효과 크기 (η²)
    grand_mean = all_data.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in group_list)
    ss_total = sum((all_data - grand_mean)**2)
    eta_sq = float(ss_between / ss_total) if ss_total > 0 else 0

    # 사후 분석 자동 연결
    posthoc_result = None
    if p < 0.05:
        try:
            if is_normal and is_equal_var:
                post = sp.posthoc_tukey(df, val_col=y_var, group_col=x_var)
                posthoc_name = 'Tukey HSD'
            elif is_normal:
                post = sp.posthoc_tamhane(df, val_col=y_var, group_col=x_var)
                posthoc_name = 'Tamhane'
            else:
                post = sp.posthoc_nemenyi(df, val_col=y_var, group_col=x_var)
                posthoc_name = 'Nemenyi'
            posthoc_result = {'method': posthoc_name, 'matrix': post.round(4).to_dict()}
        except Exception as e:
            posthoc_result = {'method': 'error', 'matrix': str(e)}

    return {
        'title': f'{x_var}에 따른 {y_var} 차이 검정',
        'question': f'{x_var} 그룹 간 {y_var}에 유의한 차이가 있는가?',
        'x_var': x_var, 'y_var': y_var,
        'h0': f'{x_var} 그룹 간 {y_var}의 평균 차이가 없다',
        'h1': f'{x_var} 그룹 간 {y_var}의 평균 차이가 있다 (최소 한 쌍)',
        'assumptions': {
            'normality': {'test': norm_test, 'statistic': float(stat_n), 'p_value': float(p_n), 'pass': bool(is_normal)},
            'equal_variance': {'test': 'Levene', 'statistic': float(stat_lev), 'p_value': float(p_lev), 'pass': bool(is_equal_var)},
        },
        'test_name': test_name,
        'statistic': float(stat), 'p_value': float(p),
        'conclusion': 'H0 기각' if p < 0.05 else 'H0 기각할 수 없음',
        'interpretation': f'그룹 간 유의한 차이가 {"있다" if p < 0.05 else "있다고 볼 수 없다"} (α=0.05)',
        'effect_size': {'name': 'η²', 'value': eta_sq},
        'posthoc': posthoc_result,
        'group_stats': {str(k): {'n': int(len(v)), 'mean': float(v.mean()), 'std': float(v.std())} for k, v in groups.items()},
    }


def analyze_correlation(df, x_var, y_var):
    """연속형 — 연속형 상관분석"""
    _df = df[[x_var, y_var]].dropna()
    n = len(_df)

    # 정규성 검정
    stat_n1, p_n1 = stats.shapiro(_df[x_var])
    stat_n2, p_n2 = stats.shapiro(_df[y_var])
    norm_test = 'Shapiro-Wilk'
    is_normal = (p_n1 > 0.05) and (p_n2 > 0.05)

    # 본 검정 자동 분기
    if is_normal:
        r, p = stats.pearsonr(_df[x_var], _df[y_var])
        test_name = 'Pearson'
    else:
        r, p = stats.spearmanr(_df[x_var], _df[y_var])
        test_name = 'Spearman'

    return {
        'title': f'{x_var}와 {y_var}의 상관분석',
        'question': f'{x_var}와 {y_var} 사이에 유의한 상관관계가 있는가?',
        'x_var': x_var, 'y_var': y_var,
        'h0': f'{x_var}와 {y_var} 사이에 유의한 상관이 없다 (ρ=0)',
        'h1': f'{x_var}와 {y_var} 사이에 유의한 상관이 있다',
        'assumptions': {
            'normality_x': {'test': norm_test, 'statistic': float(stat_n1), 'p_value': float(p_n1), 'pass': bool(p_n1 > 0.05)},
            'normality_y': {'test': norm_test, 'statistic': float(stat_n2), 'p_value': float(p_n2), 'pass': bool(p_n2 > 0.05)},
        },
        'test_name': test_name,
        'statistic': float(r), 'p_value': float(p),
        'conclusion': 'H0 기각' if p < 0.05 else 'H0 기각할 수 없음',
        'interpretation': f'유의한 상관이 {"있다" if p < 0.05 else "있다고 볼 수 없다"} (α=0.05)',
        'effect_size': {'name': 'r²', 'value': float(r**2)},
        'posthoc': None,
        'group_stats': {'n': int(n), 'r': float(r)},
    }


def analyze_chi_square(df, x_var, y_var):
    """범주형 — 범주형 카이제곱 검정"""
    ct = pd.crosstab(df[x_var], df[y_var])
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    n = int(ct.sum().sum())
    k = min(ct.shape) - 1
    cramers_v = float(np.sqrt(chi2 / (n * k))) if k > 0 else 0

    return {
        'title': f'{x_var}와 {y_var}의 독립성 검정',
        'question': f'{x_var}와 {y_var}는 독립적인가?',
        'x_var': x_var, 'y_var': y_var,
        'h0': f'{x_var}와 {y_var}는 독립적이다',
        'h1': f'{x_var}와 {y_var}는 연관성이 있다',
        'assumptions': {'min_expected': float(expected.min())},
        'test_name': '카이제곱 검정',
        'statistic': float(chi2), 'p_value': float(p),
        'conclusion': 'H0 기각' if p < 0.05 else 'H0 기각할 수 없음',
        'interpretation': f'두 변수 간 유의한 연관성이 {"있다" if p < 0.05 else "있다고 볼 수 없다"} (α=0.05)',
        'effect_size': {'name': "Cramér's V", 'value': cramers_v},
        'posthoc': None,
        'group_stats': {'dof': int(dof), 'n': n},
    }


# ── 분석 실행 ──
results = []

print("\n[1/11] PLoan에 따른 Income 차이 검정...")
results.append(analyze_two_group(df, 'PLoan', 'Income', 0, 1))

print("[2/11] PLoan에 따른 CCAvg 차이 검정...")
results.append(analyze_two_group(df, 'PLoan', 'CCAvg', 0, 1))

print("[3/11] PLoan에 따른 Age 차이 검정...")
results.append(analyze_two_group(df, 'PLoan', 'Age', 0, 1))

print("[4/11] PLoan에 따른 Experience 차이 검정...")
results.append(analyze_two_group(df, 'PLoan', 'Experience', 0, 1))

print("[5/11] PLoan에 따른 Mortgage 차이 검정...")
results.append(analyze_two_group(df, 'PLoan', 'Mortgage', 0, 1))

print("[6/11] Family 그룹에 따른 PLoan 차이 검정...")
results.append(analyze_multi_group(df, 'Family', 'PLoan'))

print("[7/11] Education 그룹에 따른 PLoan 차이 검정...")
results.append(analyze_multi_group(df, 'Education', 'PLoan'))

print("[8/11] PLoan × CDAccount 독립성 검정...")
results.append(analyze_chi_square(df, 'PLoan', 'CDAccount'))

print("[9/11] PLoan × HasMortgage 독립성 검정...")
results.append(analyze_chi_square(df, 'PLoan', 'HasMortgage'))

print("[10/11] Age × Experience 상관분석 (다중공선성 확인)...")
results.append(analyze_correlation(df, 'Age', 'Experience'))

print("[11/11] Income × CCAvg 상관분석...")
results.append(analyze_correlation(df, 'Income', 'CCAvg'))

# ── 결과 저장 ──
output = {
    'data_info': {'path': DATA_PATH, 'rows': int(df.shape[0]), 'cols': int(df.shape[1])},
    'alpha': 0.05,
    'results': results,
    'summary': [
        {
            'title': r['title'],
            'test': r['test_name'],
            'p_value': r['p_value'],
            'conclusion': r['conclusion'],
            'effect_size': r['effect_size'],
        }
        for r in results
    ],
}

OUTPUT_PATH = 'stat_results.json'
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"\n결과 저장 완료: {OUTPUT_PATH}")
