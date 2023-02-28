import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split



split_size = 0.2
folder="../hist_data"

    
def load_df(symbol): 
    path = os.path.join(folder, symbol, symbol+".csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df
    else: 
        print(f"ERROR: Couldn't find the path: {path} to load the dataframe..")
        return None
    
def store_data(train_df, test_df, symbol):
    
    # store training data as df
    train_df_path = os.path.join(folder, symbol, "train")
    if not os.path.exists(train_df_path):
        os.mkdir(train_df_path)
    train_df.to_csv(os.path.join(train_df_path, symbol+".csv"))
    
    test_df_path = os.path.join(folder, symbol, "test")
    if not os.path.exists(test_df_path):
        os.mkdir(test_df_path)
    test_df.to_csv(os.path.join(test_df_path, symbol+".csv"))

        
def split_data(symbols): 
    for symbol in symbols: 
        df = load_df(symbol)
        if df is not None: 
            train_df, test_df = train_test_split(df, test_size=split_size, shuffle=False)
            print(f"INFO: Splitted data for {symbol}: total len: {len(df)}, train_df len: {len(train_df)}, test_df len: ^{len(test_df)}")
            store_data(train_df, test_df, symbol)
        else: 
            exit(0)

if __name__ == "__main__":
    symbols = ["ADAUSDT", "SOLUSDT", "DOTUSDT", "XRPUSDT"]
    split_data(symbols)