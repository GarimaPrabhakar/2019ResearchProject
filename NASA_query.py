import os
import pandas


def query_nasa():
    """

    :return: database queried from NASA Exoplanet Archive in csv "stellar_targets.csv". Change to USB file.
    """

    base_url = 'https://exoplanetarchive.ipac.caltech.edu' \
               '/cgi-bin/nstedAPI/nph-nstedAPI?table=compositepars&' \
               'select=fpl_hostname,fpl_name,fpl_discmethod,fpl_orbper,' \
               'fpl_eccen,fpl_tranflag,fpl_cbflag,fst_optmag,fst_optmagband,' \
               'fst_spt,fst_teff'

    # do I need to create a new file path before? Check when debugging.
    os.system('wget "' + str(base_url) + '" -O "/Volumes/SFdata2018/NASAtargets/stellar_targets.csv"')


Target_file_path = '/Volumes/SFdata2018/NASAtargets/stellar_targets.csv'

query_nasa()

def initialize_col(dataframe, col_name, col_content):
    """
    Initializer for yes/no column.

    :param dataframe: a data frame
    :param col_name: the initialized column name
    :param col_content: the string to input to initialize column
    :return: the dataframe with column appended.
    """

    df = dataframe
    df_length = len(df.iloc[:, 0])
    new_col = []

    for x in range(0, df_length):
        new_col.append(col_content)

    new_dict = {str(col_name): new_col}

    df2 = pandas.DataFrame(new_dict)

    df.join(df2)

    return df


def add_yn_column(single_star_dataframe, nasa_target_df):
    """
    Check to see if a row is from a planet.
    :param single_star_dataframe: a total dataframe for one star.
    :param nasa_target_df: "star_targets.csv"
    :return: dataframe with modified yes/no column
    """

    ssdf = single_star_dataframe
    ntdf = nasa_target_df
    target_name = ssdf.iloc[1, 0]

    ssdf = initialize_col(ssdf, 'True/False', 'No')
    planet_period = []
    # target_names = list(ntdf.iloc[:,0])

    for num in range(0, len(ntdf.iloc[:, 0])):

        if str(ntdf.iloc[int(num), 0]) == str(target_name):
            planet_period = planet_period.append(ntdf.iloc[int(num), 3])

            for row_num in range(0, len(ssdf.iloc[:, 0])):
                for period in planet_period:

                    row_period = float(ssdf.iloc[row_num, :1])

                    if row_period-1.50 < period < row_period+1.50:

                        ssdf.iloc[row_num, 1] = 'Yes'

                        break

    return ssdf
