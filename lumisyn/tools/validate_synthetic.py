import pandas as pd
import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "framework"))
from acfm_net.pipeline import ACFMNetV5Pipeline

np.random.seed(42)
n = 768
df = pd.DataFrame({
    'Pregnancies':    np.random.randint(0, 18, n),
    'Glucose':        np.random.randint(70, 200, n).astype(float),
    'BloodPressure':  np.random.randint(50, 120, n).astype(float),
    'SkinThickness':  np.random.randint(10, 60, n).astype(float),
    'Insulin':        np.random.randint(0, 500, n).astype(float),
    'BMI':            np.round(np.random.uniform(18.0, 50.0, n), 1),
    'DiabetesPedigreeFunction': np.round(np.random.uniform(0.05, 2.5, n), 3),
    'Age':            np.random.randint(21, 81, n),
    'Outcome':        np.random.randint(0, 2, n),
})

pipe = ACFMNetV5Pipeline()
df_syn, report = pipe.fit_generate(df, num_samples=200, target_col='Outcome', epochs=3)

print()
print('=== SYNTHETIC DATA VALIDATION ===')

results = []

def chk(cond, label):
    status = 'PASS' if cond else 'FAIL'
    results.append(cond)
    print(f'  [{status}] {label}')
    return cond

chk((df_syn['Pregnancies'] >= 0).all(),      'Pregnancies >= 0 (no negatives)')
chk((df_syn['Pregnancies'] % 1 == 0).all(),  'Pregnancies are integers (no fractions)')
chk((df_syn['Age'] >= 0).all(),              'Age >= 0 (no negatives)')
chk((df_syn['Age'] % 1 == 0).all(),          'Age is integer (no fractions)')
chk(df_syn['Outcome'].isin([0, 1]).all(),    'Outcome in {0,1} only')
chk((df_syn['Outcome'] >= 0).all(),          'Outcome >= 0 (no negatives)')
chk((df_syn['Outcome'] <= 1).all(),          'Outcome <= 1 (no values > 1)')
chk((df_syn['Glucose'] >= 70).all() and (df_syn['Glucose'] <= 200).all(), 'Glucose within original bounds')
chk((df_syn['BMI'] >= 18.0).all() and (df_syn['BMI'] <= 50.0).all(), 'BMI within original bounds')

print()
print('Score: %d/%d checks passed' % (sum(results), len(results)))
print('Outcome distribution:', df_syn['Outcome'].value_counts().to_dict())
print('Pregnancies min/max:', df_syn['Pregnancies'].min(), '/', df_syn['Pregnancies'].max())
print('Age min/max:', df_syn['Age'].min(), '/', df_syn['Age'].max())
print()
print('Sample rows (Pregnancies, Age, Outcome):')
print(df_syn[['Pregnancies','Age','Outcome']].head(8).to_string())
