# 2019ResearchProject
Main files used in 2019 research project to investigate orbital parameter (period, eccentricity, and amplitude) relations between exoplanet and noise signals for exoplanets published at the NASA Exoplanet Archive.

This project involved three parts:
1. Calculating the period, eccentricity, and amplitude values for each periodogram signal (noise or exoplanet) from sample set. This was automated with AsciiReader.py, Feature_extraction.py, Nasa_Query.py, spectra2radvel_ceres.py, signal_period.py, data_handling.py, spec_props.py, and radvel_analysis_routine.py.
2. Comparing the period, eccentricity, and amplitude values for exoplanets versus noise signals for statistically significant exoplanet-only or noise-only regions.
3. Based on exoplanet-only or noise-only regions identified in Part 2, a classification algorithm was developed to use the false alarm probability and the regions from part 2 to classify periodogram signals as noise or exoplanet. This was done with noise_reduction.py.

Ultimately, I found three noise-only regions, as well as a strong correlation between planet multiplicity and average eccentricity. However, I discussed my results with 
Dr, Raphaelle Haywood at the University of Exeter and found that the noise-only regions are more likely due to selection bias whend designing exoplanet detection
surveys, rather than intrinsic properties of the sample. Thus, they cannot be used as a classification tool. My results and report can be found here: https://drive.google.com/drive/folders/19Rv9Sk9YohvZgV3ec3KaapHmPHcu5YCS?usp=sharing

