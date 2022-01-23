"""
This is where functions for general use are put.
Many functions from other libraries are renamed for use and ease.
"""

from astropy.io import fits
import pandas
import numpy as np
import signal
import time


def open_fits_file(fits_file_path):
    """
    This is a renaming of the fits.open file for ease of use.
    The only main change is that memmap=True.

    :param fits_file_path: must be a fits file
    :return: The opened fits file
    """
    return fits.open(fits_file_path, memmap=True)


def fits_content_summary(fits_file):
    """
    This is a one-function use of the info() function to get
    high level summaries of fits datasets.
    It is changed for ease so that the with statement for
     opening and closing of the summary data is automated.

    :param fits_file: must be an open-able fits file 
    (already opened with open_fits_file)
    :return: the content summary of the fits file
    """
    with fits_file as content_summary:
        return content_summary.info()


def fits2df(file_path):
    """
    Makes pandas dataframe from a .fits filepath.
    :param file_path: .fits filepath
    :return: filled dataframe
    """

    df = fits.open(file_path)


    with fits.open(file_path) as data:
        df = pandas.DataFrame(data[0].data)

    return df


def fits2df_ceresfits_columns(file_path):
    """
    imports ceres spectral analysis routine to a dataframe and changes its column names to the ones given below.
    :param file_path: a .fits filepath
    :return: filled dataframe
    """

    with fits.open(file_path) as data:
        df = pandas.DataFrame(data[0].data)

        pandas.DataFrame(data[0].data).columns("Wavelength", "Extracted flux", "Error extracted flux",
                                               "Blaze corrected flux", "Error in blaze corrected flux",
                                               "Continuum normalized flux", "Error in normalized flux",
                                               "Estimated continuum", "S/N ratiio", "Normalized flux*d(wavelength",
                                               "Error of 9th entrance")

        return df


def empty_df_ceresfits():

    """
    create an empty dataframe with the columns given in the .fits ceres spectral analysis routine.
    :return: empty dataframe
    """

    ceres_fits_cols = ("Wavelength", "Extracted flux", "Error extracted flux", "Blaze corrected flux",
                       "Error in blaze corrected flux",
                       "Continuum normalized flux", "Error in normalized flux",
                       "Estimated continuum", "S/N ratiio", "Normalized flux*d(wavelength",
                       "Error of 9th entrance")
    df = pandas.DataFrame(columns=ceres_fits_cols)

    return df


def txt2df(file_path):
    """
    imports the .txt and creates a dataframe for it. Columns not specified.
    :param file_path: a .txt filepath, seperated with a tab
    :return: dataframe
    """

    data = pandas.read_csv(file_path, sep="\t")
    return data


def get_nth_key(dictionary, n=0):
    """
    From:https://stackoverflow.com/questions/4326658/how-to-index-into-a-dictionary

    :param dictionary:
    :param n: the index for key number you want
    :return: the key
    """
    if n < 0:
        n += len(dictionary)
    for i, key in enumerate(dictionary.keys()):
        if i == n:
            return key
    raise IndexError("dictionary index out of range")


def csv2df(csv_filepath, cols=None):
    """
    import csv data delimited by commas into a dataframe.
    :param cols: column names in a list
    :param csv_filepath: a csv filepath. Data is delimited by commas.
    :return: dataframe
    """
    if cols is None:

        df = pandas.DataFrame(pandas.read_csv(csv_filepath))
        return df

    if cols != list:
        print('Cols not a list! Try again.')

    else:

        df = pandas.DataFrame(pandas.read_csv(csv_filepath, usecols=cols))
        return df


def delete_high_fap(dataframe):

    """
    Delete periods whose false alarm probability is high.
    :param dataframe: dataframe of periods, considering that fap is at column index 2.
    :return: modified dataframe
    """

    df = dataframe
    faps = df.iloc[:, 2]
    for num, fap in enumerate(faps, start=0):

        if float(fap) > 7.00:  # Taken from VanderPlas (2017)'s guide to the LS periodogram

                df = df.drop(num)
            # df.drop(df.index[num-1])  # drop the row with the high fap

    return df


def average(floatlist):

    """
    finds average given a list of numbers.

    :param floatlist: a list of numbers
    :return: average
    """

    return sum(floatlist)/len(floatlist)


def define_epochs(floatlist, num_epochs):

    """
    define epochs intervals given a time list and a number of epochs.

    :param floatlist: a list of dates, times
    :param num_epochs: number of epochs
    :return: a list of even time intervals
    """

    epochs = list(np.linspace(min(floatlist), max(floatlist), num=num_epochs))

    return epochs


def spect_prop_epochs(spectral_prop_dataframe, num_epochs=4):  # Not debugged yet: we need the actual specprop df

    """
    Input dataframe as:

    1. Target name
    2. Time (BJD)

    :param spectral_prop_dataframe: a df with the spectral properties
    :param num_epochs: number of epochs
    :return: a df with the spectral properties in epochs
    """

    time = spectral_prop_dataframe.iloc[:, 1]
    spdf = spectral_prop_dataframe

    epochs = list(np.linspace(min(time), max(time), num=num_epochs+1))
    epoch_data = []
    for index in range(1, len(epochs)+1):
        for ind, row in spdf.pandas.DataFrame.iterrows():
            bs = []
            bs_err = []
            eff_temp = []
            vsini = []
            mag_flux_slope = []
            mag_flux_y_int = []
            s_index = []
            haca_flux = []
            s_n_ratio = []

            if epochs[index-1] < row[1] < epochs[index]:
                bs.append(row[2])
                bs_err.append(row[3])
                eff_temp.append(row[4])
                vsini.append(row[5])
                mag_flux_slope.append(row[6])
                mag_flux_y_int.append(row[7])
                s_index.append(row[8])
                haca_flux.append(row[9])
                s_n_ratio.append(row[10])

            elif row[1] > epochs[index]:

                epoch_data.append(average(bs))
                epoch_data.append(average(bs_err))
                epoch_data.append(average(eff_temp))
                epoch_data.append(average(vsini))
                epoch_data.append(average(mag_flux_slope))
                epoch_data.append(average(mag_flux_y_int))
                epoch_data.append(average(s_index))
                epoch_data.append(average(haca_flux))
                epoch_data.append(average(s_n_ratio))

                bs.clear()
                bs_err.clear()
                eff_temp.clear()
                vsini.clear()
                mag_flux_slope.clear()
                mag_flux_y_int.clear()
                s_index.clear()
                haca_flux.clear()
                s_n_ratio.clear()

        col_names_keys = []
        col_iter_num = num_epochs

        for x in range(1, col_iter_num+1):  # num_epochs is also the number of iterations over the parameters.
            col_names_keys.append('Avg Bisector Span: Epoch ' + str(x))
            col_names_keys.append('Avg Bisector Span Error: Epoch ' + str(x))
            col_names_keys.append('Avg Effective Temperature: Epoch' + str(x))
            col_names_keys.append('Avg Vsini: Epoch ' + str(x))
            col_names_keys.append('Avg Stellar Magnetic Flux: y-intercept: Epoch ' + str(x))
            col_names_keys.append('Avg Stellar Magnetic Flux: slope: Epoch ' + str(x))
            col_names_keys.append('Avg S_Index: Epoch ' + str(x))
            col_names_keys.append('Avg Hα/Ca Flux index: Epoch ') + str(x)
            col_names_keys.append('Avg Signal to Noise Ratio: Epoch ' + str(x))

        spec_prop_dict = dict(zip(col_names_keys, epoch_data))

        spec_prop_df = pandas.DataFrame(spec_prop_dict)
        return spec_prop_df


class GracefulKiller:
  kill_now = False
  def __init__(self):
    signal.signal(signal.SIGINT, self.exit_gracefully)
    signal.signal(signal.SIGTERM, self.exit_gracefully)

  def exit_gracefully(self,signum, frame):
    self.kill_now = True
