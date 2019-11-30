def get_folds(K, Xy_train):
    '''
    :param K: No. of folds
    :param Xy_train: X_train and y_train still concatenated  
    :return: number of folds, 
    '''
    fold_size = round(len(Xy_train) / K)  # rounding down, the remainder will be dealt with in the final column of df_rand
    df_rand = pd.DataFrame(pd.np.empty([fold_size + (K - 1), K]) * pd.np.nan)  # large dataframe containing rand_nums
    df_folds = pd.DataFrame()  # we are going to add each fold i.e. df_fold as a column to this
    # df_rand is essentially a mapping where each cell contains the rand_num telling us to look at X_train[rand_num, :]
    arr = []
    for j in range(0, len(Xy_train)):
        arr.append(j)  # creating array that contains numbers that represent indices/rows of X_train
    for k in range(0, K - 1):  # inputting data into first K-1 rows of df_rand because they are all the same size
        df_fold = pd.DataFrame(pd.np.empty([fold_size, 1]) * pd.np.nan).astype(
            object)  # think of df_fold as a filing cabinet, where each cell will contain an array that is a row of X_train
        for i in range(0, fold_size):
            rand_ind = random.randint(0, len(arr) - 1)  # -1 from len(arr) to agree with indexing
            rand_num = arr[rand_ind]  # get the number at the index of arr
            df_rand.iloc[i, k] = rand_num  # put the number into df_rand
            df_fold.at[i, 0] = np.array(
                Xy_train.iloc[int(df_rand.iloc[i, k])])  # put the rand_numth row from X_train into df_fold cell
            arr.pop(
                rand_ind)  # remove the rand_num (referenced by rand_ind) to ensure we do not pick the same X_train row in a later fold
        df_folds = pd.concat([df_folds, df_fold], axis=1)
    # Now to deal with the Kth fold - this may have a different length to the initial K-1 folds...
    # So we create a dataframe with the remaining numbers in arr, access the corresponding rows in X_train and put into last column of df_rand (no need to use df_fold for the last fold)
    arr_df = pd.DataFrame(arr)
    df_rand.iloc[:, K - 1] = arr_df
    for i in range(0, len(arr_df)):
            df_fold.at[i, 0] = np.array(Xy_train.iloc[int(arr_df.iloc[i, 0])])
    df_folds = pd.concat([df_folds, df_fold], axis=1)  # df_fold here is the Kth df_fold that we just created withing the loop
    return K, df_folds


def cross_fold_val(K, algo, df_folds):  # K-fold cross validation
    '''
    Our own cross fold validation function that performs K folding iterations across X_train and
    applies algo each time. We wrote our own so that we could return the folds and access them after
    just in case the metrics exhibited large variations on different folding iterations
    :param K: no of folds
    :param algo: 
    :param df_folds: folds passed from get_folds()
    :return:
    '''
    df_folds.columns = list(range(len(df_folds.columns)))
    df_conmat = pd.DataFrame(pd.np.empty([K, 4]) * pd.np.nan)  # this will be the dataframe of confusion matrices where each row corresponds to each folding iteration's (K's) confusion matrix
    df_conmat.columns = ['TruePos', 'TrueNeg', 'FalseNeg', 'FalsePos']
    # Creating total dataframes to return at end of fn
    X_train_fold_total = pd.DataFrame()
    y_train_fold_total = pd.DataFrame()
    X_test_fold_total = pd.DataFrame()
    y_test_fold_total = pd.DataFrame()
    log_time_fold_total = []
    # y_pred_fold_total = pd.DataFrame()  # no need to return this for now, as df_conmat uses y_preds_fold in it's calculations already each folding iteration
    logloss_fold_total = []
    r2_fold_total = []
    for i in range(0, K):  # no.of folds K
        df_folds_copy = df_folds  # each time we change folding set up (loop through i), we create a copy of initial wall of filing cabinets
        test_fold = df_folds_copy.iloc[:, i].to_frame()  # one particular filing cabinet (i) held out for testing purposes. Making it a daraframe as this allows everything to work properly below this point
        x_test_fold = test_fold
        y_test_fold = pd.DataFrame(pd.np.empty([len(test_fold), 1]) * pd.np.nan).astype(object)  # creating df to populate in j loop
        for j in range(len(test_fold.dropna())):  # looping through each drawer(aka array)in filing cabinet (nan check is necessary, because the last filing cabinet (fold) is longer and dictates the size of the other filing cabinets, thus the first K-1 folds have at least one nan at the bottom) Note: len(test_fold[~test_fold.isna()]) won't work for some reason:
            y_test_fold.iloc[j, 0] = test_fold.iloc[j, 0][0]  # add the very first file (element) in each drawer (array) to y_test_fold
            x_test_fold.iloc[j, 0] = np.delete(x_test_fold.iloc[j, 0], 0)  # remove the first file from each drawer to yield x_test_fold
        # Do similar for train_fold (outside of j loop, back in i loop)
        train_fold = df_folds_copy.drop(df_folds_copy.index[i], axis=1)  # dropping the test filing cabinet (i) from the wall. Using this for 1) size of x_train_fold and 2) also maybe when fitting models
        train_fold_melted = train_fold.melt().dropna().value.to_frame()  # collapsing all columns into one long col, not sure if I should be dropping nan's right now
        train_fold_melted.reset_index(drop=True, inplace=True)  # have to reset the indices as they didn't update after dropping nans
        x_train_fold = train_fold_melted
        y_train_fold = pd.DataFrame(pd.np.empty([len(train_fold_melted), 1]) * pd.np.nan).astype(object)  # creating df to populate in j loop
        for j in range(len(train_fold_melted)):
            y_train_fold.iloc[j, 0] = train_fold_melted.iloc[j, 0][0]  # add the very first file (element) in each drawer (array) to y_test_fold
            x_train_fold.iloc[j, 0] = np.delete(train_fold_melted.iloc[j, 0], 0)  # remove the first file from each drawer to yield x_test_fold
        # need to place each element in array into its own cell of a new df, so that it can be fit
        X_train_fold = pd.DataFrame(pd.np.empty([len(x_train_fold), len(x_train_fold.iloc[0, 0])]) * pd.np.nan)  # 0,0 is not there for any specific reason
        X_test_fold = pd.DataFrame(pd.np.empty([len(x_test_fold), len(x_test_fold.iloc[0, 0])]) * pd.np.nan)
        for drawer in range(len(X_train_fold)):
            for file in range(len(x_train_fold.iloc[drawer, 0])):
                X_train_fold.iloc[drawer, file] = x_train_fold.iloc[drawer, 0][file]  # X_train_fold is the finished training df
        for drawer in range(len(x_test_fold.dropna())):
            for file in range(len(x_test_fold.iloc[drawer, 0])):
                X_test_fold.iloc[drawer, file] = x_test_fold.iloc[drawer, 0][file]
        X_test_fold.dropna(inplace=True)  # X_test_fold is the finished testing df
        y_test_fold.dropna(inplace=True)  # need to drop nans to avoid issues when fitting
        # fitting algo on currently defined folds
        model = model_fit_predict(algo, X_train_fold, y_train_fold, X_test_fold)
        log_time_fold = model[0]
        y_pred_fold = model[1]
        log_time_fold_total.append(log_time_fold)  # appending within the loop as it is fit on each fold within each folding iteration i.e. (K-1)*K values
        # working on confusion matrix
        df_conmat.iloc[i, :] = manual_confusion_matrix(y_test_fold, y_pred_fold)  # sanity check with sklearn fn: confusion_matrix(y_test_fold, y_preds)
        # appending each dataframe to total dataframes
        X_train_fold_total = pd.concat([X_train_fold_total, X_train_fold], axis=1)
        y_train_fold_total = pd.concat([y_train_fold_total, y_train_fold], axis=1)
        X_test_fold_total = pd.concat([X_test_fold_total, X_test_fold], axis=1)
        y_test_fold_total = pd.concat([y_test_fold_total, y_test_fold], axis=1)
        # y_preds_fold_total = pd.concat([y_preds_fold_total, y_preds_fold], axis=1)
        # running log_loss on each folding iteration and concatenating into one dataframe
        logloss_fold = log_loss(y_test_fold, algo.predict_proba(X_test_fold)[:, 1])
        logloss_fold_total.append(logloss_fold)
        # running r squared on each folding iteration and concatenating into one dataframe
        # r2_fold = r2_score(y_test_fold, algo.predict_proba(X_test_fold)[:, 1])
        # r2_fold_total.append(r2_fold)
    return df_conmat, X_train_fold_total, y_train_fold_total, X_test_fold_total, y_test_fold_total, K, logloss_fold_total, np.mean(log_time_fold_total) # r2_fold_total , y_preds_fold_total
