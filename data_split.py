from sklearn.model_selection import train_test_split
import pandas as pd

import warnings
warnings.filterwarnings(action='ignore')

files = ['LDAP','MSSQL','Syn','NetBIOS','Portmap','UDP','UDPLag']

print('-----------------------------------------------------------------')

for file in files :
    f_name = file+'.csv'
    train = file+'_train.csv'
    test = file+'_test.csv'

    # Load the data
    df = pd.read_csv(f_name)
    df_train = pd.read_csv(train)
    df_test = pd.read_csv(test)

    print('Load " ',f_name,'" is completed')
    #to check data shape
##    print(f_name,': ',df.shape)
##    print()
##    print(train,': ',df_train.shape)
##    print()
##    print(test,': ',df_test.shape)
##    print()
    
    #split data
    X_train, X_test = train_test_split(df, test_size = 0.3, random_state = 123)
    print('Data is split')
    print()
    #save each data ( csv file )
    X_train.to_csv(train)
    X_test.to_csv(test)

    print('store',train,'and',test,' is completed')
    print()
    print('-----------------------------------------------------------------')
    
    





