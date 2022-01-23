"""
Automated pipeline to download and analyze spectra through ESO archive, given a list of target names.
The code for this is modified from http://archive.eso.org/cms/faq/how-do-i-programmatically-access-the-archive.html

Computes: (in txt file)
0-  Object name
1-  BJD
2-  RV
3-  error in RV
4-  Bisector span
5-  error in bisector span
6-  instrument
7-  pipeline
8-  resolving power
9-  Effective Temperature
10- log(g)
11- [Fe/H]
12- v*sin(i)
13- value of the continuum normalized ccf at it lowest point
14- standard deviation of the gaussian fitted to the ccf
15- Exposure time
16- Signal to noise ratio at ~5150 \AA
17- path to the ccf plot file

Along with calibrated/reduced spectra with flux, wavelength, uncertainty, etc., and ccf plots
Using ceres at https://github.com/rabrahm/ceres.

Automated terminal script is used for ease.
"""

import os
from subprocess import call as ccall
import data_handling
import pandas

your_username = "Gprab"
your_password = "Gp12345$"
Instrument = 'HARPS'
FileCat = 'SCIENCE'
target_names = list(['HD147513', 'HD100623'])
Max_rows = 200


def raw_data_query_fp(list_target_names=target_names):
    file_paths = []
    var = 'HD147513'
    for one_target_name in list_target_names:

        file_path = os.system("/Volumes/SFDATA2018/ScienceFair2018_code/prog_access.sh -User <"
                  + str(your_username) + "> -Passw <" +
                  str(your_password) + "> -FileCat " + str(FileCat) + " -Inst " + str(Instrument) + " -TargetName "
                  + str(var)) # + " -Max_rows 200"

        print(file_path)

        # file_path = "/Volumes/SFdata2018/" + str(one_target_name)
        # # + "/" + os.environ("reqnum"), IF USING THIS REMEMBER TO EXPORT $REQNUM ON prog_access.sh
        #
        # files4dir = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
        #
        # # Another option found on Stack Overflow: import glob, print(glob.glob("/home/adam/*.txt"))
        #
        # file_paths.append(files4dir)
        for fp in file_path:
            filep = data_handling.fits2df(fp)
            if len(filep.columns) == 3:
                file_paths.append(fp)

    return file_paths




def raw_data_analysis(raw_data_filepaths):
    """

    :param raw_data_filepaths: a list of lists of raw_data
    (each list within list is the filepaths to the raw data for each target).

    :return:

    - All rv txt file paths in a list of list,
    - all calibrated/reduced spectra file paths in a list of a list,
    - all ccf plots file paths in a list of lists,
    - rv txt directory file paths in a list,
    - calibrated/reduced spectra directory file paths in a list,
    - and ccf plots directory in a list
    """
    os.system("cd /Volumes/SFDATA2018/ceres")
    os.system("python HARPSpipe.py")

    rv_dir_file_paths = []
    calibrated_spectra_dfp = []
    ccf_plots_dfp = []

    txt_dirs = []
    fits_dirs = []
    pdf_dirs = []

    for target_name in target_names:
        for file_dir in raw_data_filepaths:

            rv_file_paths = []
            calibrated_spectra_fp = []
            ccf_plots_fp = []

            new_dir_txt = str(target_name) + "_red_txt"
            txt_dirs.append(new_dir_txt)

            new_dir_fits = str(target_name) + "_red_fits"
            fits_dirs.append(new_dir_fits)

            new_dir_pdf = str(target_name) + "_red_pdf"
            pdf_dirs.append(new_dir_pdf)

            os.system("mkdir " + str(new_dir_txt))
            os.system("mkdir " + str(new_dir_fits))
            os.system("mkdir " + str(new_dir_pdf))

            for file in file_dir:
                # Generates spectra, rvs/attributes, and ccf plots
                os.system("python harpspipe.py " + str(file) + str(" -do_class -npools 2 -dirout "))
                # The above line actually does all the cooking ;)

                #os.system("cd /Volumes/SFdata2018")

                rv_file = str(file) + "_red.txt"
                calibrated_spectra = str(file) + "_red.fits"
                ccf_plot = str(file) + "_red.pdf"

                rv_file_paths.append(rv_file)
                calibrated_spectra_fp.append(calibrated_spectra)
                ccf_plots_fp.append(ccf_plot)

                os.system("mkdir " + str(file) + "_red_txt")
                # Some problem here with "no file found" at this location...?
                os.system("mkdir " + str(file) + "_red_fits")
                os.system("mkdir " + str(file) + "_red_pdf")

                os.system("cd /Volumes/SFDATA2018/ceres")


            rv_dir_file_paths.append(rv_file_paths)
            calibrated_spectra_dfp.append(calibrated_spectra_fp)
            ccf_plots_dfp.append(ccf_plots_fp)

    return rv_dir_file_paths, calibrated_spectra_dfp, ccf_plots_dfp, txt_dirs, fits_dirs, pdf_dirs

def raws_data_analysis(spectra_file_paths):
    """
    The function raw_data_analysis doesn't seem to be doing the right thing, so I made another test function.
    Outputs all filepaths outputed by ceres in a list, by iterating over each file given
    :param spectrafilepaths: list
    A list of filpaths to the spectra of the star.

    :return: list
    fits, txt and pdf files which have been run through the ceres harps pipeline.
    """
    let_us_see = []
    for file in spectra_file_paths:

        if file == []:
            print("Uh oh, this file isn't a string! Skipping...")
            print(file)

        if file == '':
            print("Uh oh, this file isn't a string! Skipping...")
            print(file)

        else:

            print(data_handling.fits2df(file))

            ccall(['cd', ' /Users/Shefali_Prabhakar/Documents/GitHub/ceres'],shell=True)
            ccall(['cd', ' harps'], shell=True)
            print("working dir:: ", str(ccall(args= 'pwd', shell= True)))

            # os.system('cd /Users/Shefali_Prabhakar/Documents/GitHub/ceres')
            # os.system('cd harps')
            print(file)
            lets_see = ccall(['python harpspipe.py ' + str(file)], shell=True)
            #lets_see = ccall(['python', 'harpspipe.py', str(file)], shell=True)
            print('Done with this part...')

            #lets_see= os.system("python harpspipe.py " + str(file) + str(" -do_class -npools 2 -dirout "))
            #print(os.system("python /Volumes/SFDATA2018/ceres/harps/harpspipe.py " + str(file) + str(" -do_class -npools 2 -dirout ")))
            print(lets_see)
            let_us_see.append(lets_see)

    return let_us_see

# print(data_handling.fits2df('/Users/Shefali_Prabhakar/Downloads/HARPS.2017-07-16T22_53_49.874.fits'))
print(raws_data_analysis(['/Users/Shefali_Prabhakar/Downloads/HARPS.2017-07-16T22_53_49.874.fits']))