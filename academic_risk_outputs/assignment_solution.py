import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import accuracy_score
import graphviz

OUT=Path(__file__).resolve().parent
raw_data = {
    'Student_ID': [1001, 1002, 1003, 1004, 1005, 1001, 1006, 1007, 1008, 1009],
    'County': ['Nakuru', 'Narok', 'Nakuru', 'Narok', 'Kakamega', 'Nakuru', 'Nairobi', None, 'Narok', 'Nakuru'],
    'Study_Hours_Per_Week': [15, 8, 20, None, 12, 15, 30, 10, 25, 18],
    'Attendance_Percent': [85, 60, 92, 55, 78, 85, 95, 65, 88, None],
    'Fee_Balance_Ksh': [0, 15000, 0, 25000, 5000, 0, 0, 18000, 2000, 12000],
    'Previous_Grade': [72, 48, 85, 45, 65, 72, 90, 50, 82, 58],
    'Student_Type': ['Regular', 'Regular', 'Scholarship', 'Regular', 'Scholarship', 'Regular', 'Regular', 'Scholarship', 'Regular', 'Regular'],
    'Academic_Risk': [0, 1, 0, 1, 0, 0, 0, 1, 0, 1]
}
df=pd.DataFrame(raw_data)
with open(OUT/'results.txt','w') as f:
 def log(*x): print(*x,file=f)
 log('A. ORIGINAL SHAPE:',df.shape)
 log('DATA TYPES:\n',df.dtypes.to_string())
 log('DESCRIBE:\n',df.describe().to_string())
 log('B. MISSING:\n',df.isna().sum().to_string())
 log('EXACT DUPLICATES:',int(df.duplicated().sum()))
 log('DUPLICATED STUDENT IDS:',int(df.Student_ID.duplicated().sum()))
 df=df.drop_duplicates(subset=['Student_ID'],keep='first').copy().reset_index(drop=True)
 df['County']=df['County'].fillna('Unknown')
 for c in ['Study_Hours_Per_Week','Attendance_Percent']:
  df[c]=df[c].fillna(df[c].median())
 log('CLEANED SHAPE:',df.shape)
 log('CLEANED DATA:\n',df.to_string(index=False))
 log('RISK COUNTS:\n',df.Academic_Risk.value_counts().sort_index().to_string())
 log('RISK GROUP MEANS:\n',df.groupby('Academic_Risk')[['Study_Hours_Per_Week','Attendance_Percent','Fee_Balance_Ksh','Previous_Grade']].mean().to_string())
 log('COUNTY RISK RATES:\n',df.groupby('County')['Academic_Risk'].agg(['count','mean']).to_string())
 df['Attendance_Category']=pd.cut(df.Attendance_Percent,bins=[-np.inf,70,85,np.inf],labels=['Low','Moderate','High'],right=False)
 df['Has_Fee_Balance']=(df.Fee_Balance_Ksh>0).astype(int)
 log('ENGINEERED:\n',df[['Student_ID','Attendance_Category','Has_Fee_Balance']].to_string(index=False))
 numeric=['Study_Hours_Per_Week','Attendance_Percent','Fee_Balance_Ksh','Previous_Grade']
 corr=df[numeric+['Has_Fee_Balance','Academic_Risk']].corr()
 log('CORRELATIONS WITH RISK:\n',corr['Academic_Risk'].sort_values().to_string())
 features=numeric+['Has_Fee_Balance']
 encoded=pd.get_dummies(df[['County','Student_Type','Attendance_Category']],dtype=int)
 X=pd.concat([df[features],encoded],axis=1)
 scaler=StandardScaler()
 X[numeric]=pd.DataFrame(scaler.fit_transform(X[numeric]),columns=numeric,index=X.index)
 y=df.Academic_Risk
 log('ENCODED/STANDARDIZED FEATURES:\n',X.round(3).to_string(index=False))
 model=DecisionTreeClassifier(max_depth=3,random_state=42)
 model.fit(X,y)
 log('TREE ROOT:',X.columns[model.tree_.feature[0]])
 log('TREE RULES:\n',export_text(model,feature_names=list(X.columns)))
 log('TRAINING ACCURACY:',accuracy_score(y,model.predict(X)))
 df.to_csv(OUT/'cleaned_student_data.csv',index=False)
 X.assign(Academic_Risk=y).to_csv(OUT/'model_ready_data.csv',index=False)
 sns.set_theme(style='whitegrid')
 fig,axs=plt.subplots(2,2,figsize=(12,9))
 for ax,col in zip(axs.flat,numeric):
  sns.boxplot(data=df,x='Academic_Risk',y=col,ax=ax,color='#7baed2')
  sns.stripplot(data=df,x='Academic_Risk',y=col,ax=ax,color='#253858',size=5)
  ax.set_xticks([0,1],['Not at risk','At risk'])
 fig.suptitle('EDA: Academic Risk and Numerical Factors',fontsize=15)
 fig.tight_layout();fig.savefig(OUT/'eda_numeric.png',dpi=160);plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(12,4.5))
 sns.countplot(data=df,x='County',hue='Academic_Risk',ax=axs[0]);axs[0].tick_params(axis='x',rotation=30);axs[0].set_title('Student Counts by County and Risk')
 sns.countplot(data=df,x='Attendance_Category',hue='Academic_Risk',ax=axs[1]);axs[1].set_title('Attendance Category and Risk')
 fig.tight_layout();fig.savefig(OUT/'eda_categories.png',dpi=160);plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,6));sns.heatmap(corr,annot=True,cmap='coolwarm',vmin=-1,vmax=1,fmt='.2f',ax=ax);fig.tight_layout();fig.savefig(OUT/'correlation_heatmap.png',dpi=160);plt.close(fig)
 dot=export_graphviz(model,out_file=None,feature_names=list(X.columns),class_names=['Not at risk','At risk'],filled=True,rounded=True,impurity=True)
 (OUT/'decision_tree.dot').write_text(dot)
 graphviz.Source(dot).render(str(OUT/'decision_tree'),format='png',cleanup=True)
 graphviz.Source(dot).render(str(OUT/'decision_tree'),format='pdf',cleanup=False)
 print((OUT/'results.txt').read_text())
