import os, glob, re, zipfile, shutil
import pandas as pd
from astropy.io import fits
from astropy.time import Time

print('--- PART 1: Concrete Verification Example ---')
# Load GOES
df = pd.read_parquet('data/raw/noaa_catalog.parquet')
df = df[df['peak_time'] >= pd.Timestamp('2023-09-01', tz='UTC')]
df = df.copy()
df['start_time'] = pd.to_datetime(df['date'] + df['start'], format='%Y%m%d%H%M', utc=True)
df['end_time'] = pd.to_datetime(df['date'] + df['end'], format='%Y%m%d%H%M', utc=True)
df.loc[df['end_time'] < df['start_time'], 'end_time'] += pd.Timedelta(days=1)
df = df.dropna(subset=['start_time', 'end_time'])
df_mx = df[df['xray_class'].str[0].isin(['M', 'X'])]

# FIND JUST ONE FILE THAT MATCHES
zips = glob.glob(r'd:\Users\Nitro 5\Downloads\HEL1OS_Data\**\HLS_*.zip', recursive=True)
pattern = re.compile(r'HLS_(\d{8})_(\d{6})_(\d+)sec_lev1_V\d+')

match_found = False
for z in zips:
    name = os.path.basename(z)
    m = pattern.match(name)
    if not m: continue
    
    d_str, t_str, dur_str = m.groups()
    start_t = pd.to_datetime(f'{d_str}{t_str}', format='%Y%m%d%H%M%S', utc=True)
    end_t = start_t + pd.Timedelta(seconds=int(dur_str))
    
    overlaps = df_mx[(start_t <= df_mx['end_time']) & (end_t >= df_mx['start_time'])]
    if not overlaps.empty:
        match_found = True
        flare = overlaps.iloc[0]
        print(f'MATCHED ZIP: {name}')
        print(f'  Filename Window : {start_t} to {end_t}')
        print(f'  Target Flare    : {flare["date"]} Class {flare["xray_class"]}, Window: {flare["start_time"]} to {flare["end_time"]}')
        
        os.makedirs('d:/tmp_fits2', exist_ok=True)
        with zipfile.ZipFile(z, 'r') as zf:
            fits_names = [n for n in zf.namelist() if n.lower().endswith('.fits') and 'cdte1' in n.lower()]
            if not fits_names: fits_names = [n for n in zf.namelist() if n.lower().endswith('.fits')]
            if fits_names:
                ext = zf.extract(fits_names[0], 'd:/tmp_fits2')
                with fits.open(ext) as hdul:
                    if len(hdul) > 1 and 'MJD' in [c.name for c in hdul[1].columns]:
                        mjd_start = hdul[1].data['MJD'][0]
                        mjd_end = hdul[1].data['MJD'][-1]
                        t_start_fits = Time(mjd_start, format='mjd', scale='utc').datetime
                        t_end_fits = Time(mjd_end, format='mjd', scale='utc').datetime
                        print(f'  FITS Header MJD : {t_start_fits} UTC to {t_end_fits} UTC')
                        delta = (pd.Timestamp(t_start_fits, tz='UTC') - start_t).total_seconds()
                        print(f'  Delta (FITS - Filename): {delta} seconds')
        break

print('\n--- PART 2: Quick Subset Extraction Test ---')
# Get 4 files manually without loading all 2570 sizes
test_zips = [
    r'd:\Users\Nitro 5\Downloads\HEL1OS_Data\2023\11\30\HLS_20231130_085955_53995sec_lev1_V111.zip',
    r'd:\Users\Nitro 5\Downloads\HEL1OS_Data\2024\01\01\HLS_20240101_000000_15951sec_lev1_V121.zip',
    r'd:\Users\Nitro 5\Downloads\HEL1OS_Data\2024\02\15\HLS_20240215_120000_7194sec_lev1_V131.zip',
    r'd:\Users\Nitro 5\Downloads\HEL1OS_Data\2024\03\10\HLS_20240310_060000_28794sec_lev1_V131.zip'
]
# Filter out non-existent ones
test_zips = [z for z in test_zips if os.path.exists(z)]
if not test_zips:
    test_zips = zips[:4] # fallback

os.makedirs('d:/hel1os_quick_test', exist_ok=True)
total_zip = 0
total_ext = 0

for i, z in enumerate(test_zips):
    dest = f'd:/hel1os_quick_test/test_{i}'
    os.makedirs(dest, exist_ok=True)
    
    with zipfile.ZipFile(z, 'r') as zf:
        zf.extractall(dest)
        
    extracted = sum(os.path.getsize(os.path.join(r, f)) for r, d, files in os.walk(dest) for f in files)
    
    comp = os.path.getsize(z)
    total_zip += comp
    total_ext += extracted
    
    print(f'Extracting: {os.path.basename(z)} ({comp/1024**2:.2f} MB)')
    print(f'  Extracted: {extracted/1024**2:.2f} MB')
    print(f'  Expansion: {extracted/comp:.2f}x')

print('\n--- Subset Results ---')
if total_zip > 0:
    print(f'Weighted Expansion for Subset Test: {total_ext/total_zip:.2f}x')
    print(f'Projected Extracted Size (assuming 51.43 GB subset): {(51.43 * (total_ext/total_zip)):.2f} GB')

shutil.rmtree('d:/tmp_fits2', ignore_errors=True)
shutil.rmtree('d:/hel1os_quick_test', ignore_errors=True)
