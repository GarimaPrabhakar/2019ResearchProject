"""
Data Set Compiler for the features.

The useful information on each spectrum (and therefore each radial velocity data point) is averaged for four epochs over
the observation length. This information is assigned to each period and corresponding keplerian
point to make the number of dimensions per data point 42.

The Data is shuffled and all properties being used in the algorithm
are compiled into one data set, and saved to a csv file.

These properties are:: signal amplitude, eccentricity, false alarm probability,
orbital phase, argument of periastron, period, bisector velocity span,
log'HK, and stellar magnetic flux

"""

import NASA_query
import spectra2radvel_ceres
import signal_period
import data_handling
import pandas as pd
import spec_props
import radvel_analysis_routine


# = input('Filepath for Output Dataset:: ')

file_path = '/Volumes/SFdata2018/Exoplanet_Training_Sets/exoplanet_training_set1.csv'

NASA_query.query_nasa()
df = data_handling.csv2df('/Volumes/SFdata2018/NASAtargets/stellar_targets.csv')

# target_names = df[0]
target_names = list('PROXIMA')

tot_results_dataframe = pd.DataFrame(columns=9)
tot_results_dataframe.to_csv(str(file_path))

for target_name in target_names:

    target_name = list(target_name) #input requires a list
    file_paths = list(list(spectra2radvel_ceres.raw_data_query_fp(target_name)))

    rv_dir_file_paths, calibrated_spectra_dfp, ccf_plots_dfp, txt_dirs, fits_dirs, pdf_dir = \
        spectra2radvel_ceres.raw_data_analysis(file_paths)

    # calibrated_spectra_dfp and rv_fir_file_paths are what we're looking for.

    radveldf = pd.DataFrame(columns=["Object Name", "BJD", "RV", "Error in RV"])
    radeveldf = pd.DataFrame(radveldf[1, 3])

    # calspecdf = pd.DataFrame(columns=["Wavelength", "Extracted_flux", "Error in Extracted Flux"])

    df1 = data_handling.empty_df_ceresfits()
    df2 = data_handling.empty_df_cerestxt()

    tot_dataset = pd.concat([df1, df2], axis=1) # the whole dataset

    for rvpath in rv_dir_file_paths:

        rvdf = data_handling.txt2df(rvpath)  # formatting the original ceres radvel routine dataframe
        rvdf.columns = range(df.shape[1])

        newdf = rvdf.drop(df.columns[13:], axis=1)
        newdf = newdf.drop(df.columns[10, 11], axis=1, inplace=True)
        newdf = newdf.drop(df.columns[6, 8], axis=1, inplace=True)

        radveldf.pd.DataFrame.append(newdf)
        df1.pd.DataFrame.append(rvdf)

    for specpath in calibrated_spectra_dfp:

        specdf = data_handling.fits2df(specpath)  # formatting the original ceres radvel routine dataframe
        specdf.columns = range(df.shape[1])

        newspecdf = specdf.drop(specdf.columns[3:], axis=1)

        radveldf.pd.DataFrame.append(newspecdf)
        df2.pd.DataFrame.append(specdf)

    kepleriandf = signal_period.kepler_params(target_name, t=radveldf[1], y=radveldf[2], err=radveldf[3])
    spectral_props = \
        spec_props.spectral_properties_df(rvfilepaths=calibrated_spectra_dfp, spectrafilepaths=rv_dir_file_paths)

    result = pd.concat([kepleriandf, spectral_props], ignore_index=True)
    ssdf = NASA_query.add_yn_column(result, df)


    radvel_analysis_routine.radvel_analysis(rv_dir_file_paths)

    tot_results_dataframe.append(result)
