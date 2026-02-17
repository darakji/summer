
import pandas as pd
import pickle

try:
    with open('Thesis_Results/Ref_A_T1_meta.pkl', 'rb') as f:
        data = pickle.load(f)
    print("Keys:", data.keys())
    if isinstance(data, pd.DataFrame):
        print("Columns:", data.columns)
        print("First row:", data.iloc[0])
except Exception as e:
    print(e)
