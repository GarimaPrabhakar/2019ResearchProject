import os


def radvel_analysis(rv_filepath):
    """
    Control.

    The basic radvel fitting routine. Will give basic statistis
    and analysis about the exoplanet it thinks is the planet dataset given the time, radial velocity, and err.

    :param rv_filepath: the rv_filepath found by ceres pipeline. No dataset transformations are needed
    (will automatically index first three columns)
    :return: nothings.
    """

    os.system('radvel fit -s ' + str(rv_filepath))
    os.system('radvel plot -t rv -s ' + str(rv_filepath))
    os.system('radvel mcmc -s ' + str(rv_filepath))

    os.system('radvel derive -s ' + str(rv_filepath))
    os.system('radvel plot -t derived -s ' + str(rv_filepath))
    os.system('radvel ic -t nplanets e trend -s ' + str(rv_filepath))

    os.system('radvel ic -t nplanets e trend -s' + str(rv_filepath))
    os.system('radvel report -s ' + str(rv_filepath))


radvel_analysis("/Volumes/SFDATA2018/Sample_data/42_Dra_sample_timeseries.csv")