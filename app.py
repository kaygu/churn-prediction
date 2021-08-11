import pandas as pd
import numpy as np

if __name__ == '__main__':
  df = pd.read_csv('BankChurners.csv')

  ''' Clean dataset '''
  df = df.iloc[:,:-2] # Remove 2 last columns
  df.drop('CLIENTNUM', axis=1, inplace=True)
  print(f'There are {df.shape[0]} rows and {df.shape[1]} columns in this dataset')
  # Tranform object dtypes to category
  #df['Dependent_count'] = df['Dependent_count'].astype('category')
  objs = df.select_dtypes(['object'])
  for col in objs.columns:
    df[col] = df[col].astype('category')
  
  cat = df.select_dtypes('category')
  ''' Show value counts for every category '''
  # for col in cat.columns:
  #   print(f'\n{col} :\n', df[col].value_counts())
  #   print('---')
  # print(f'\nChurn rate : {len(df[df["Attrition_Flag"] == "Attrited Customer"]) / df.shape[0] * 100:.3f}%')
  # print('---')
  # print(df.describe().T)
  
  ''' Show Stacked Plots (Attrition Flag + Category) '''
  # for col in cat.columns:
  #   stacked_plot(df[col])
 
  '''
    VISUALIZATION
  '''
  import matplotlib.pyplot as plt
  import seaborn as sns

  def stacked_plot(x: pd.Series):
    '''
      Stack categorical values (x) with 'Attrition Flag'
    '''
    sns.set(palette='Set1')
    tab1 = pd.crosstab(x, df['Attrition_Flag'], margins=True)
    print(tab1)
    print('-'*50)
    tab = pd.crosstab(x, df['Attrition_Flag'], normalize='index')
    tab.plot(kind='bar', stacked=True, figsize=(10,5))
    plt.legend(loc='lower left', frameon=True)
    plt.legend(loc="upper left", bbox_to_anchor=(1,1))
    plt.ylabel('Percentage')
    plt.show()

  ''' Display correlation matrix '''
  # corr= df.corr()
  # plt.figure(figsize=(15,10))
  # sns.heatmap(corr,annot= True,vmin=-0.5,vmax=1, cmap='coolwarm',linewidths=0.75)
  # plt.show()

  ''' Total_Trans_Ct vs Total_Trans_Amt '''
  # plt.figure(figsize=(15,7))
  # sns.lineplot(x= 'Total_Trans_Ct', y='Total_Trans_Amt', hue='Attrition_Flag', data=df)
  # plt.show()

  ''' ScatterPlot '''
  # plt.figure(figsize=(15,7))
  # sns.scatterplot(x='Total_Ct_Chng_Q4_Q1', y='Total_Amt_Chng_Q4_Q1',hue='Attrition_Flag', data=df)
  # plt.show()

  ''' JointPlot '''
  # plt.figure(figsize=(15,7))
  # sns.jointplot(x= 'Avg_Utilization_Ratio',y='Total_Revolving_Bal', hue = 'Attrition_Flag', data=df)
  # plt.show()


  '''
    MODEL
  '''
  from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
  from sklearn.pipeline import make_pipeline
  from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.metrics import confusion_matrix, classification_report
  from sklearn.preprocessing import MinMaxScaler
  from imblearn.over_sampling import SMOTE

  X= df.drop(['Attrition_Flag'],axis=1)
  X =pd.get_dummies(X ,drop_first=True) # Split categories to dummies
  Y = df['Attrition_Flag'].apply(lambda x: x=='Attrited Customer').astype('int') # Converts Attrition_Flag to boolean then to int
  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=13, stratify=Y)
  
  # RESAMPLE
  ov_sample= SMOTE(random_state = 13)
  X_train, y_train = ov_sample.fit_resample(X_train, y_train)

  #pipeline
  model = make_pipeline(MinMaxScaler(), GradientBoostingClassifier(random_state=13))
  
  # Evaluate model
  cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=13)
  cv_scores = cross_val_score(model, X, Y, scoring='accuracy', cv=cv, n_jobs=-1)
  print(f'Accuracy: {np.mean(cv_scores):.3f} ({np.std(cv_scores):.3f})')

  model.fit(X_train, y_train)
  print(f'Accuracy score (training): {model.score(X_train, y_train):.3f}')
  print(f'Accuracy score (validation): {model.score(X_test, y_test):.3f}')
  y_pred = model.predict(X_test)
  # print(confusion_matrix(y_test, y_pred))
  print(classification_report(y_test, y_pred))
  # print(model.predict_proba(X_test)) # Gives probability of result
