import glob
from astropy.io import fits
from astropy.time import Time
import pandas as pd

# Let's read the extracted file
ext_paths = glob.glob('d:/quick_ext/HLS_20231130_235953_28794sec_lev1_V111.zip/**/*_lightcurve_cdte1.fits', recursive=True)
if ext_paths:
    ext_path = ext_paths[0]
    with fits.open(ext_path) as hdul:
        mjd_start = hdul[1].data['MJD'][0]
        mjd_end = hdul[1].data['MJD'][-1]
        t_start = Time(mjd_start, format='mjd', scale='utc').datetime
        t_end = Time(mjd_end, format='mjd', scale='utc').datetime
        print(f'MJD start UTC: {t_start}')
        print(f'MJD end UTC: {t_end}')
