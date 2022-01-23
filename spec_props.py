import Ha_line
import S_index
import stellar_magnetic_flux
import data_handling
import pandas


def spectral_properties(spectradataframe, infodataframe):

        """
        Input data frame has columns in order:
        - 0"Wavelength",
        - 1"Extracted flux",
        - 2"Error extracted flux",
        - 3"Blaze corrected flux",
        - 4"Error in blaze corrected flux"
        - 5"Continuum normalized flux",
        - 6"Error in normalized flux",
        - 7"Estimated continuum",
        - 8"S/N ratiio",
        - 9"Normalized flux*d(wavelength",
        - 10"Error of 9th entrance"

        COLUMNS BELOW ONLY HAVE ONE ROW. FIX THIS.
        - 0"Object Name",
        - 2"BJD",
        - 3"RV",
        - 4"Error in RV",
        - 5"Bisector Span",
        - 6"Error in bisector span",
        - 7"Instrument",
        - 8"Pipeline",
        - 9"Resolving power",
        - 10"Effective temperature",
        - 11"Log(g)",
        - 12"[Fe/H]",
        - 13"v*sini",
        - 14"Value of the lowest continuum normalized point (CCF)",
        - 15"Std of the gaussian fitted to CCF",
        - 16"Exposure time",
        - 17"S?N ratio at ~5150/AA",
        - 18"path to CCF plot file",

        Output is only one row.

        :param spectradataframe: a dataframe
        :param infodataframe: a dataframe
        :return: Bisector Velocity Span, Error in Bisector Velocity Span, Effective Temperature,
        Vsini, Stellar Magnetic Flux: y-intercept, Stellar Magnetic Flux: slope, S_Index,
        Hα/Ca Flux index, Signal to Noise Ratio.
        """

        wavelength_df = spectradataframe.iloc[0, 2]
        infodf = infodataframe

        bs = infodf.iloc[5]
        bs_err = infodf.iloc[6]
        teff = infodf.iloc[10]
        vsini = infodf.iloc[13]
        mag_flux_yint = pandas.DataFrame()
        mag_flux_slope = pandas.DataFrame()
        ss_index = pandas.DataFrame()
        Hak_flux = pandas.DataFrame()
        sn_ratio = pandas.DataFrame()

        slope, y_int = stellar_magnetic_flux.autocorrelation_analysis(wavelength_df)
        s_index = S_index.fp2s_index(wavelength_df)
        hak_flux = Ha_line.ha_flux(wavelength_df)
        avg_sn_ratio = data_handling.average(wavelength_df[8])

        mag_flux_slope.append(slope)
        mag_flux_yint.append(y_int)
        ss_index.append(s_index)
        Hak_flux.append(hak_flux)
        sn_ratio.append(avg_sn_ratio)

        return bs, bs_err, teff, vsini, mag_flux_yint, mag_flux_slope, ss_index, Hak_flux, sn_ratio


def spectral_properties_df(rvfilepaths, spectrafilepaths):
    """

    :return: Output dataframe has columns in order: Bisector Velocity Span, Error in Bisector Velocity Span,
    Effective Temperature, Vsini, Stellar Magnetic Flux: y-intercept, Stellar Magnetic Flux: slope, S_Index,
    Hα/Ca Flux index, Signal to Noise Ratio.
    """
    spectral_dataframe = pandas.DataFrame(columns=9)

    rvfps = rvfilepaths
    specfps = spectrafilepaths

    for rvfp, specfp in rvfps, specfps:

        radvdf = data_handling.txt2df(rvfp)
        spetrdf = data_handling.txt2df(specfp)

        bs, bs_err, teff, vsini, mag_flux_yint, \
            mag_flux_slope, ss_index, \
            Hak_flux, sn_ratio = \
            spectral_properties(spectradataframe=spetrdf, infodataframe=radvdf)

        spectral_props = [bs, bs_err, teff, vsini, mag_flux_yint, mag_flux_slope, ss_index, Hak_flux, sn_ratio]

        spectral_dataframe.append(spectral_props)

    data = data_handling.spect_prop_epochs(spectral_dataframe, num_epochs=4)

    return data
