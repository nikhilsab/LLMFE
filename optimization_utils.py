import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pandas.api.types import is_string_dtype


def is_categorical(x):
    assert type(x) == pd.Series
    x = x.convert_dtypes()
    if is_string_dtype(x):
        return True
    
    elif set(x) == {0, 1}:
        return True
    
    elif x.dtype in [int, float, 'Int64', 'Float64']:
        return False
    
    else:
        return True
    
    
def fill_missing(X_train_org, X_test_org):
    X_without_missings = X_train_org.copy().dropna()
    categorical_indicator_org = [is_categorical(X_without_missings.iloc[:, i]) for i in range(X_without_missings.shape[1])]
    for col_idx, col_name in enumerate(X_train_org.columns):
        if categorical_indicator_org[col_idx] == True:
            # Convert to string and handle missing values for categorical columns
            X_train_org[col_name] = X_train_org[col_name].astype('string').fillna('missing')
            X_test_org[col_name] = X_test_org[col_name].astype('string').fillna('missing')
        else:
            # Handle missing values and infinities for numerical columns in X_train_org
            X_train_org[col_name] = X_train_org[col_name].fillna(0)
            train_val_array = np.array(X_train_org[col_name].values)
            X_train_org[col_name] = np.nan_to_num(train_val_array, nan=0)
            
            # Replace infinities
            X_train_org[col_name] = X_train_org[col_name].replace({np.inf: 0, -np.inf: 0})
            
            # Convert to float
            X_train_org[col_name] = X_train_org[col_name].astype('float')

            # Handle missing values and infinities for numerical columns in X_test_org
            X_test_org[col_name] = X_test_org[col_name].fillna(0)
            test_val_array = np.array(X_test_org[col_name].values)
            X_test_org[col_name] = np.nan_to_num(test_val_array, nan=0)
            
            # Replace infinities
            X_test_org[col_name] = X_test_org[col_name].replace({np.inf: 0, -np.inf: 0})
            
            # Convert to float
            X_test_org[col_name] = X_test_org[col_name].astype('float')

            
    for k, v in X_train_org.dtypes.to_dict().items():
        X_test_org[k] = X_test_org[k].astype(v)

    return X_train_org, X_test_org


def _load_data(rule_dir, dataset, seed, split=-1):
    if split >= 0:
        saved_file_name = f'./LLM_results/{rule_dir}/function-{dataset}-{seed}-{split}.out'
    else:
        saved_file_name = f'./LLM_results/{rule_dir}/function-{dataset}-{seed}.out'
    assert(os.path.isfile(saved_file_name))

    with open(saved_file_name, 'r') as f:
        total_fct_str = f.read().strip()
        fct_strs = total_fct_str.split("\n\n---DIVIDER---\n\n")
        
    return fct_strs


def add_data_features(
    _DATA, _SEED, _RULE_DIR, X_train, X_test
):
    fct_strs = _load_data(_RULE_DIR, _DATA, _SEED)
    print(f'Number of Response: {len(fct_strs)}')
    
    total_nc = 0
    X_train_new_cols = []
    X_test_new_cols = []
    X_train_org = X_train.copy()
    X_test_org = X_test.copy()
    
    for fct_idx, fct_str_handled in enumerate(fct_strs):       
        exec(fct_str_handled)
        X_train_new_col = locals()['column_appender'](X_train_org)
        X_test_new_col = locals()['column_appender'](X_test_org)

        train_new_columns = set(X_train_new_col.columns) - set(X_train_org.columns)
        test_new_columns = set(X_test_new_col.columns) - set(X_test_org.columns)
        new_columns = train_new_columns & test_new_columns
        total_nc += len(new_columns)

        X_train_new_col = X_train_new_col[new_columns]
        X_train_new_col.columns = [f'{col_name}_{fct_idx}' for col_name in new_columns]

        X_test_new_col = X_test_new_col[new_columns]
        X_test_new_col.columns = [f'{col_name}_{fct_idx}' for col_name in new_columns]

        X_train.reset_index(drop=True, inplace=True)
        X_train_new_col.reset_index(drop=True, inplace=True)
        X_test.reset_index(drop=True, inplace=True)
        X_test_new_col.reset_index(drop=True, inplace=True)
        
        if X_train_new_col.shape[0] != X_train.shape[0]:
            continue

        assert(X_train_new_col.shape[1] == X_test_new_col.shape[1])
        assert(set(X_train_new_col.columns) == set(X_test_new_col.columns))
        
        X_train_new_cols.append(X_train_new_col)
        X_test_new_cols.append(X_test_new_col)
    
    X_train_new = pd.concat(X_train_new_cols, axis=1)
    X_test_new = pd.concat(X_test_new_cols, axis=1)
    
    print(f'Number of New Raw Columns: {total_nc}')
    print(f'Number of New Helpful Columns: {X_train_new.shape[1]}')
    
    return X_train_new, X_test_new


def filter_features(X_train, X_test):
    X_total = pd.concat([X_train, X_test], axis=0)
    
    X_total.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_total_filtered = X_total.T.drop_duplicates().T
    X_total_filtered = X_total_filtered.dropna(axis=1)
    
    selected_list = []
    for i, tr_column in enumerate(X_total_filtered.columns):
        if X_total_filtered[tr_column].nunique() == 1:
            continue
            
        if X_train[tr_column].nunique() == 1:
            continue
        
        try:
            c = X_total_filtered[tr_column].convert_dtypes()
        except:
            continue

        selected_list.append(tr_column)
            
    X_train_filtered = X_train[selected_list]
    X_test_filtered = X_test[selected_list]

    assert(X_test_filtered.shape[1] == X_train_filtered.shape[1])

    return X_train_filtered, X_test_filtered


def process(X, categorical_indicator, enc=None, scaler=None):
    categorial_indices = np.where(np.array(categorical_indicator) == True)[0]
    
    if len(categorial_indices) > 0:
        X_cat = X.iloc[:, [*categorial_indices]].astype('string')
        # cat_columns = [x for x, y in zip(X.columns.tolist(), categorial_indices) if y == 'True']
        # X_norm = pd.get_dummies(X, columns=cat_columns, drop_first=True)
        if enc == None:
            enc = OneHotEncoder(handle_unknown='ignore')     
            X_cat_new = pd.DataFrame(enc.fit_transform(X_cat).toarray())
        else:
            X_cat_new = pd.DataFrame(enc.transform(X_cat).toarray())
        X_cat_new = X_cat_new.values
        noncat_indices = np.where(np.array(categorical_indicator) == False)[0]
        
        if len(noncat_indices) > 0:
            X_noncat = X.iloc[:, [*noncat_indices]]
            if scaler == None:
                scaler = StandardScaler()
                X_noncat = scaler.fit_transform(X_noncat)
            else:
                X_noncat = scaler.transform(X_noncat)
                
            X_norm = np.concatenate([X_noncat, X_cat_new], axis=1)
        else:
            X_norm = X_cat_new
    else:
        #  X_norm = X
        if scaler == None:
            scaler = StandardScaler()
            X_norm = scaler.fit_transform(X)
        else:
            X_norm = scaler.transform(X)
            
    return X_norm, enc, scaler