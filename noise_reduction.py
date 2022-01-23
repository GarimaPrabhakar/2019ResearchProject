import pandas as pd
import collections
import os


# Some general purpose functions:

def fp2df(filepath):
    """
    imports data based on filepath in csv format
    :param filepath: string (csv filepath)
    :return: pandas dataframe
    """

    dataframe = pd.read_csv(filepath, encoding='latin-1', error_bad_lines=False)

    return dataframe


# Noise reduction functions:

def dp_ranking(dataframe):
    """
    ranks datapoints in terms of highest power/lowest FAP

    :param dataframe: pandas data frame (labelled)
    :return: ranked data frame
    """

    datafram = dataframe.sort_values(by=['Power', 'FAP'], ascending=False, inplace=False)

    return datafram


def per_eccen_amp_noise_reduction(dataframe, add_column=False):
    """
    proposed noise treatment of data: based on empirical period and eccentricity analysis
    :param dataframe: pandas data frame (labelled)
    :param add_column: boolean
    :return: one pandas data frame (with an extra (labelled)
    column: 0 for predicted noise, 1 for predicted planet)
    """

    exoamp_avg = 10.70754521  # empirically found average amplitude (exoplanet)
    exoamp_stdev = 39.31326512  # empirically found average standard deviation (exoplanet)
    dfseries = []
    predf = dataframe
    if 'Unnamed: 0.1' in predf.columns:
        del predf['Unnamed: 0.1']
    if 'Unnamed: 0.1.1' in predf.columns:
        del predf['Unnamed: 0.1.1']
    if 'Unnamed: 0.1.1.1' in predf.columns:
        del predf['Unnamed: 0.1.1.1']
    if 'Unnamed: 0.1.1.1.1' in predf.columns:
        del predf['Unnamed: 0.1.1.1.1']
    if 'Unnamed: 0.1.1.1.1.1' in predf.columns:
        del predf['Unnamed: 0.1.1.1.1.1']

    for num in range(0, len(dataframe)):
        dfseries.append(1)

    datafram = dataframe
    # print(datafram)

    if add_column:
        datafram['Predicted Results: PEA treatment'] = pd.Series(dfseries)

    for num in range(0, len(dataframe)):

        if dataframe.iloc[num, 2] <= 351.4:
            if dataframe.iloc[num, 5] >= 0.7628:  # empirically found period/ecentricity bounds
                print('Bingo!: 1')
                datafram.iloc[num, 8] = 0

        if dataframe.iloc[num, 7] >= float(exoamp_avg + exoamp_stdev + 20):
            print('Bingo!: 2')
            datafram.iloc[num, 8] = 0

        if dataframe.iloc[num, 7] <= (exoamp_avg - exoamp_stdev - 20):
            print('Bingo!: 3')
            datafram.iloc[num, 8] = 0

    return datafram


def control_reduction(dataframe):
    """
    control noise treatment of data
    :param dataframe: pandas data frame (labelled)
    :return: one pandas data frame (with an extra (labelled)
    column: 0 for predicted noise, 1 for predicted planet)
    """

    datafram = dp_ranking(dataframe)
    datafram['Predicted Results'] = pd.Series()

    for y in range(0, len(datafram)):
        if y <= 15:
            datafram.iloc[y, 8] = 1

        else:
            datafram.iloc[y, 8] = 0

    return datafram


# Accuracy metrics:

def correct_planet_hunter(exofilepath, star_id):
    """
    identifies correct planet from validated exoplanet dataset (input: filepath)
    given its (previously) assigned ID number
    :param exofilepath: string
    :param star_id: int
    :return: dataframe with indexed (correct) exoplanets
    """

    exodf = fp2df(exofilepath)

    dataframe = pd.DataFrame()

    for y in range(0, len(exodf)):

        if exodf['Name'][y] == star_id:
            dataframe = dataframe.append(exodf.iloc[y])

    return dataframe


def confusion_matrix(pred_dataframe, corr_dataframe):
    """
    outputs number of: False Positives, False Negatives, True Positives,
    True Negatives in an ordered dictionary and the data frame
    with an additional (labelled) column with 0: as ACTUAL noise and 1: as ACTUAL planet

    :param pred_dataframe: pandas data frame (predictions dataset)
    :param corr_dataframe: pandas data frame (with the datapoints of correct exoplanets)
    :return: OrderedDict, dataframe
    """

    predf = pred_dataframe
    # print(corr_dataframe)
    listo = []
    for y in range(0, len(predf)):
        listo.append(0)

    predf['Actual Results'] = pd.Series(listo)

    # print(corr_dataframe)
    predf = pd.DataFrame(predf)

    if 'Unnamed: 0.1' in predf.columns:
        del predf['Unnamed: 0.1']
    if 'Unnamed: 0.1.1' in predf.columns:
        del predf['Unnamed: 0.1.1']
    if 'Unnamed: 0.1.1.1' in predf.columns:
        del predf['Unnamed: 0.1.1.1']
    if 'Unnamed: 0.1.1.1.1' in predf.columns:
        del predf['Unnamed: 0.1.1.1.1']
    if 'Unnamed: 0.1.1.1.1.1' in predf.columns:
        del predf['Unnamed: 0.1.1.1.1.1']
    # print(predf)

    for z in range(0, len(corr_dataframe)):  # for every row in the correct df
        for y in range(0, len(predf)):  # for every row in the predf

            if corr_dataframe.iloc[z, 0] == predf.iloc[y, 1]:  # if the star numbers are the same,
                if corr_dataframe.iloc[z, 0] == predf.iloc[y, 0]:  # if the rperiod number and row # are the same
                    predf.iloc[y, 9] = 1  # append 1 to the correct location

    # print(predf)
    fp = []
    fn = []
    tp = []
    tn = []

    for y in range(0, len(predf)):

        if predf.iloc[y].iloc[8] == 1 and predf.iloc[y].iloc[9] == 1:
            tp.append(1)

        if predf.iloc[y, 8] == 1 and predf.iloc[y].iloc[9] == 0:
            fp.append(1)

        if predf.iloc[y, 8] == 0 and predf.iloc[y].iloc[9] == 0:
            tn.append(1)

        if predf.iloc[y, 8] == 0 and predf.iloc[y].iloc[9] == 1:
            fn.append(1)

        # if predf.loc[predf.index[y]][9] == 0 and predf.loc[predf.index[y]][9] == 1:
        #         fn.append(1)

    cmatrix = collections.OrderedDict({'False Positives': len(fp), 'False Negatives': len(fn),
                                       'True Positives': len(tp), 'True Negatives': len(tn)})

    print('CONFUSION MATRIX:: ')
    print('False Positives:: ' + str(len(fp)))
    print('False Negatives:: ' + str(len(fn)))
    print('True Positive:: ' + str(len(tp)))
    print('True Negative:: ' + str(len(tn)))

    return cmatrix, predf


def accuracy(fp, fn, tp, tn):
    """
    returns model accuracy from number of: false_positive, false_negative, true_positive, true_negative
    :param fp: int
    :param fn: int
    :param tp: int
    :param tn: int
    :return: float
    """

    corr_preds = tp + tn
    tot_preds = fp + fn + tp + tn

    accuracy_ = (corr_preds / tot_preds) * 100
    print('ACCURACY:: ' + str(accuracy_))

    return accuracy_


def auc(fp, fn, tp, tn):
    """
    returns model auc from number of: false_positive, false_negative, true_positive, true_negative

    NOTE: first element outputted is false positive rate, next element in true positive rate
    :param fp: int
    :param fn: int
    :param tp: int
    :param tn: int
    :return: float, float
    """
    fpr = 0
    tpr = 0
    if fp == 0 and (fp + tn) == 0:
        fpr = 'Nan'

    elif tp == 0 and (fn + tp) == 0:
        tpr = 'Nan'

    else:

        fpr = fp / (fp + tn)
        tpr = tp / (fn + tp)

        print('FALSE POSITIVE RATE:: ' + str(fpr))
        print('TRUE POSITIVE RATE:: ' + str(tpr))

    return fpr, tpr


def f1(fp, fn, tp):
    """
    returns model precision, recall, and F1 score from number of:
    false_positive, false_negative, true_positive, true_negative

    NOTE: outputs are returned as follows: precision, recall, F1 score

    :param fp: int
    :param fn: int
    :param tp: int
    :return: float
    """

    precision = 0
    recall = 0
    f1_score = 'NAN'
    if tp == 0 and (tp + fp) == 0:
        precision = 'Nan'

    elif tp == 0 and (fn + tp) == 0:
        recall = 'Nan'

    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        if precision == 0 or recall == 0:
            f1_score = 'Nan'

        else:
            f1_score = 2 * (1 / ((1 / precision) + (1 / recall)))

        print('PRECISION:: ' + str(precision))
        print('RECALL:: ' + str(recall))
        print('F1 SCORE:: ' + str(f1_score))

    return precision, recall, f1_score


# The actual Bang for all that Buck! :


def control_noise_reduction(keparams_filepath, star_ID,
                            correct_exo_filpath='/Volumes/SFDATA2018/Total Exoplanet Data/other exodatasets/Exoplanets_Dataset_authentic csv copy.csv'):
    """
    executes control trial of experiment, utilising only power/FAP bounds and accuracy analysis.
    outputs: data frame with accuracy results

    NOTE:: two additional (labelled) columns are added to the keparams_filepath.

    :param correct_exo_filpath: string (path to validated exoplanet dataset)
    :param star_ID: string
    :param keparams_filepath: string (path to computed keplerian parameters)
    :return: pandas data frame
    """

    keparamsdf = fp2df(keparams_filepath)

    keparamsdf = control_reduction(keparamsdf)

    correct_df = correct_planet_hunter(correct_exo_filpath, star_ID)

    cmatrix, kepdf = confusion_matrix(pred_dataframe=keparamsdf, corr_dataframe=correct_df)

    fp = cmatrix['False Positives']
    tp = cmatrix['True Positives']
    fn = cmatrix['False Negatives']
    tn = cmatrix['True Negatives']

    acc = accuracy(fp=fp, fn=fn, tp=tp, tn=tn)
    fpr, tpr = auc(fp=fp, fn=fn, tp=tp, tn=tn)
    precision, recall, f1_ = f1(fp=fp, fn=fn, tp=tp)

    anadict = collections.OrderedDict({'Star #': star_ID, 'FP': list([fp]), 'FN': list([fn]), 'TP': list([tp]),
                                       'TN': list([tn]), 'Accuracy': list([acc]),
                                       'FPR': list([fpr]), 'TPR': list([tpr]), 'Precision': list([precision]),
                                       'Recall': list([recall]), 'F1 Score': list([f1_])})

    datafram = pd.DataFrame.from_dict(anadict)

    return datafram


def goat_planet_spotter(keparams_filepath, star_ID,
                        correct_exo_filepath='/Volumes/SFDATA2018/Total Exoplanet Data/other exodatasets/Exoplanets_Dataset_authentic csv copy.csv'):
    """
     executes proposed noise reduction trial of experiment,
     utilising only period/eccentricity/amplitude bounds and accuracy analysis.
     outputs: data frame with accuracy results

     NOTE:: two additional (labelled) columns are added to the keparams_filepath.

     :param correct_exo_filepath: string (path to validated exoplanet dataset)
     :param keparams_filepath: string (path to computed keplerian parameters)
     :param star_ID: int
     :return: pandas data frame
     """

    keparamsdf = fp2df(keparams_filepath)

    # keparamsdf = per_eccen_amp_noise_reduction(keparamsdf)
    keparamsdf = control_reduction(keparamsdf)
    keparams_df = per_eccen_amp_noise_reduction(keparamsdf)
    correct_df = correct_planet_hunter(correct_exo_filepath, star_ID)

    cmatrix, kepdf = confusion_matrix(pred_dataframe=keparams_df, corr_dataframe=correct_df)

    fp = cmatrix['False Positives']
    tp = cmatrix['True Positives']
    fn = cmatrix['False Negatives']
    tn = cmatrix['True Negatives']

    acc = accuracy(fp=fp, fn=fn, tp=tp, tn=tn)
    fpr, tpr = auc(fp=fp, fn=fn, tp=tp, tn=tn)
    precision, recall, f1_ = f1(fp=fp, fn=fn, tp=tp)

    anadict = collections.OrderedDict({'Star #': star_ID, 'FP': list([fp]), 'FN': list([fn]), 'TP': list([tp]),
                                       'TN': list([tn]), 'Accuracy': list([acc]),
                                       'FPR': list([fpr]), 'TPR': list([tpr]), 'Precision': list([precision]),
                                       'Recall': list([recall]), 'F1 Score': list([f1_])})

    datafram = pd.DataFrame.from_dict(anadict)

    return datafram


# def the_og_planet_spotter(keparams_filepath,
#                           star_ID, correct_exo_filepath='/Volumes/SFDATA2018/Total Exoplanet Data/other exodatasets/Exoplanets_Dataset_authentic csv copy.csv'):
def the_og_planet_spotter(keparams_filepath,
                          star_ID,
                          correct_exo_filepath='/Users/Shefali_Prabhakar/Downloads/Exoplanets_Dataset_authentic csv copy.csv'):
    """
     executes combined proposed noise reduction trial of experiment,
     utilising period/eccentricity/amplitude bounds, power/FAP bounds, and accuracy analysis.
     outputs: data frame with accuracy results

     NOTE:: two additional (labelled) columns are added to the keparams_filepath.

     :param correct_exo_filepath: string (path to validated exoplanet dataset)
     :param keparams_filepath: string (path to computed keplerian parameters)
     :param star_ID: int
     :return: pandas data frame
     """
    keparamsdf = fp2df(keparams_filepath)

    predf = keparamsdf

    if 'Unnamed: 0.1' in predf.columns:
        del predf['Unnamed: 0.1']
    if 'Unnamed: 0.1.1' in predf.columns:
        del predf['Unnamed: 0.1.1']
    if 'Unnamed: 0.1.1.1' in predf.columns:
        del predf['Unnamed: 0.1.1.1']
    if 'Unnamed: 0.1.1.1.1' in predf.columns:
        del predf['Unnamed: 0.1.1.1.1']
    if 'Unnamed: 0.1.1.1.1.1' in predf.columns:
        del predf['Unnamed: 0.1.1.1.1.1']

    keparamsdf = per_eccen_amp_noise_reduction(keparamsdf, add_column=True)

    correct_df = correct_planet_hunter(correct_exo_filepath, star_ID)

    cmatrix, kepdf = confusion_matrix(pred_dataframe=keparamsdf, corr_dataframe=correct_df)

    pd.DataFrame.to_csv(kepdf, keparams_filepath)

    fp = cmatrix['False Positives']
    tp = cmatrix['True Positives']
    fn = cmatrix['False Negatives']
    tn = cmatrix['True Negatives']

    acc = accuracy(fp=fp, fn=fn, tp=tp, tn=tn)
    fpr, tpr = auc(fp=fp, fn=fn, tp=tp, tn=tn)
    precision, recall, f1_ = f1(fp=fp, fn=fn, tp=tp)

    anadict = collections.OrderedDict({'Star #': star_ID, 'FP': list([fp]), 'FN': list([fn]), 'TP': list([tp]),
                                       'TN': list([tn]), 'Accuracy': list([acc]),
                                       'FPR': list([fpr]), 'TPR': list([tpr]), 'Precision': list([precision]),
                                       'Recall': list([recall]), 'F1 Score': list([f1_])})

    datafram = pd.DataFrame.from_dict(anadict)

    return datafram


# star_lists = [1,7,8,10,12,13,22,24,30,33,34,38,40,43,56,58,59,68,69,77,80,82,84,87,
#               90,105,107,108,114,116,119,122,123,125,134,136,138,139,140,141,144,149,
#               151,152,156,160,165,172,174,175,176,177,179,183,184,188,192,196,198]

starIDs = pd.read_csv("CorrectExoplanetsHiRES2.csv")["Name"].values.tolist()

andict = collections.OrderedDict(
    {'Star#': pd.Series(), 'FP': pd.Series(), 'FN': pd.Series(), 'TP': pd.Series(), 'TN': pd.Series(),
     'Accuracy': pd.Series(),
     'FPR': pd.Series(), 'TPR': pd.Series(), 'Precision': pd.Series(), 'Recall': pd.Series(), 'F1 Score': pd.Series()})

df = pd.DataFrame.from_dict(andict)

for x in starIDs:
    dfd = control_noise_reduction(keparams_filepath=('DATA_real2/' +
                                                     str(x) + '.csv'), star_ID=x,
                                  correct_exo_filpath="CorrectExoplanetsHiRES2.csv")

    # dfd = the_og_planet_spotter(keparams_filepath=('DATA_real2/' +
    #                str(x) + '.csv'), star_ID=x)
    df = df.append(dfd)

print(df)

df.to_csv("control_results1.csv")
