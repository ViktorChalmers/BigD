def split_data(data, n=10):
    X_train=np.zeros([round(len(data)/n),n])

    X_test=[]
    y_train=[]
    y_test=[]
    '''
    for i in range(n):
        X_train[i], X_test[i], y_train[i], y_test[i] = train_test_split(X, y, train_size=1/n, test_size=1/n)
    '''


    return X_train, X_test, y_train, y_test