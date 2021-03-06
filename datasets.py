import pandas as pd
import numpy as np



def read_real_dataset(dims_3 = False):

    ''' read real dataset from home_data.csv '''

    #             y             x1
    columns = ['price', 'sqft_living']

    if dims_3:     #x2   
        columns += ['sqft_above']

    data = pd.read_csv('./dataset/home_data.csv').loc[:, columns]
    # data.hist(bins=150) # check dataset distribution
    data=(data-data.mean())/data.std() # standartization
    x, y = data.iloc[:, 1:].to_numpy(), data.iloc[:,0].to_numpy()
    x_0 = np.ones(len(x))
    x = np.column_stack((x_0, x))

    #x_2 = np.random.normal(0, 1, x.shape[0])#np.square(x) * np.sin(x/10)
    #x = np.column_stack((x_0, x, x_2))
    
    x_test, y_test = x[20000:], y[20000:]

    return x, y, x_test, y_test



def generate_dataset(w_1, w_0, noise = False):

    ''' generate toy noise dataset '''

    x = np.arange(-10, 10, 0.3) # generate X
    x_0 = np.ones(len(x))
    x = np.column_stack((x_0, x))


    y_func = lambda x: np.matmul(x, np.array([w_0, w_1]))
    
    y_ideal = np.array([y_func(x) for x_vec in x])
    y_ideal = y_ideal.astype('float64')

    if noise:
        y_noised = y_ideal + np.random.normal(0, 1, len(y_ideal))

    return x, y_ideal, y_noised