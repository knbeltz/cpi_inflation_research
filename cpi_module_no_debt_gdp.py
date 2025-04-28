import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt  
import statsmodels.api as sm 
from sklearn.feature_selection import f_regression, SelectKBest
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import pearsonr 

df_inflation = pd.read_excel('data/dependent_data/imf_inflation_rate.xls')

df_cpi = pd.read_excel('data/cpi_data/CPI_2012-2023.xlsx')
df_m2 = pd.read_csv(r'data/other_independent_data/wb_broad_money_indicator_%gdp.csv')
df_gdp_growth = pd.read_csv(r'data/other_independent_data/wb_gdp_growth%.csv')
df_gfcf = pd.read_csv(r'data/other_independent_data/wb_gross_fixed_capital_formation_%gdp.csv')
df_debt_gdp = pd.read_excel('data/other_independent_data/imf_debt_gdp_ratio.xls')
df_unemployment = pd.read_excel('data/other_independent_data/wb_unemployment_rate.xls')

def full_analysis(year): 
  def merge_data():
    global df_inflation, df_cpi, df_m2, df_gdp_growth, df_gfcf, df_debt_gdp, df_unemployment

    df_cpi = df_cpi.rename(columns={'Country ': 'Country'})
    df_m2 = df_m2.rename(columns={'Country ': 'Country'})
    df_gdp_growth = df_gdp_growth.rename(columns={'Country ': 'Country'})
    df_gfcf = df_gfcf.rename(columns={'Country ': 'Country'}) 
    df_debt_gdp = df_debt_gdp.rename(columns={'Country ': 'Country'})
    df_unemployment = df_unemployment.rename(columns={'Country ': 'Country'})

    merged_df = pd.merge(df_inflation[['Country', f'{year} inflation rate %']], df_cpi[['Country', f'CPI {year}']], on='Country', how='left')
    merged_df = pd.merge(merged_df, df_m2[['Country', f'Broad money (% of GDP) {year}']])
    merged_df = pd.merge(merged_df, df_gdp_growth[['Country', f'GDP growth (annual %) {year}']])
    merged_df = pd.merge(merged_df, df_gfcf[['Country', f'Gross fixed capital formation (% of GDP) {year}']])
    merged_df = pd.merge(merged_df, df_unemployment[['Country', f'{year} unemployment rate %']], on='Country', how='left')

    merged_df = merged_df[merged_df[f'{year} inflation rate %'] < 250]
    return merged_df
  
  merged_df = merge_data()
    
  def seaborn_plot():
    sns.lmplot(x=f'CPI {year}', y=f"{year} inflation rate %", data=merged_df)
    plt.show()

  def seaborn_cc(): 
    subset = merged_df[[f'CPI {year}', f"{year} inflation rate %"]].dropna()
    cc = pearsonr(subset[f'CPI {year}'], subset[f"{year} inflation rate %"])
    return cc
  
  def VIF():
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print(vif_data)

  def train_test_X(): 
    merged_df_clean = merged_df.dropna()
    X = merged_df_clean[[f'CPI {year}', f'Broad money (% of GDP) {year}', f'GDP growth (annual %) {year}', f'{year} unemployment rate %']]
    return X

  def train_test_y():
    merged_df_clean = merged_df.dropna()
    y = merged_df_clean[f'{year} inflation rate %']
    return y
  
  X = train_test_X()
  y = train_test_y()

  def OLS_info():
    selector = SelectKBest(score_func=f_regression, k='all')
    X_new = selector.fit_transform(X, y)
    model_final = sm.OLS(y, sm.add_constant(X_new)).fit()
    print(model_final.summary())
  
  print(seaborn_plot())
  print(seaborn_cc())
  print(VIF())
  print(OLS_info())
