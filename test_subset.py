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

zips = glob.glob(r'd:\Users\Nitro 5\Downloads\HEL1OS_Data\**\HLS_*.zip', recursive=True)
pattern = re.compile(r'HLS_(\d{8})_(\d{6})_(\d+)sec_lev1_V\d+')

hls_records = []
match_found = False

for z in zips:
    name = os.path.basename(z)
    m = pattern.match(name)
    if not m: continue
    
    d_str, t_str, dur_str = m.groups()
    start_t = pd.to_datetime(f'{d_str}{t_str}', format='%Y%m%d%H%M%S', utc=True)
    end_t = start_t + pd.Timedelta(seconds=int(dur_str))
    
    hls_records.append({
        'filepath': z, 'filename': name,
        'start_time': start_t, 'end_time': end_t,
        'size': os.path.getsize(z)
    })
    
    if not match_found:
        overlaps = df_mx[(start_t <= df_mx['end_time']) & (end_t >= df_mx['start_time'])]
        if not overlaps.empty:
            match_found = True
            flare = overlaps.iloc[0]
            print(f'MATCHED ZIP: {name}')
            print(f'  Filename Window : {start_t} to {end_t}')
            print(f'  Target Flare    : {flare["date"]} Class {flare["xray_class"]}, Window: {flare["start_time"]} to {flare["end_time"]}')
            
            os.makedirs('d:/tmp_fits', exist_ok=True)
            with zipfile.ZipFile(z, 'r') as zf:
                fits_names = [n for n in zf.namelist() if n.lower().endswith('.fits') and 'cdte1' in n.lower()]
                if not fits_names: fits_names = [n for n in zf.namelist() if n.lower().endswith('.fits')]
                if fits_names:
                    ext = zf.extract(fits_names[0], 'd:/tmp_fits')
                    with fits.open(ext) as hdul:
                        if len(hdul) > 1 and 'MJD' in [c.name for c in hdul[1].columns]:
                            mjd_start = hdul[1].data['MJD'][0]
                            mjd_end = hdul[1].data['MJD'][-1]
                            t_start_fits = Time(mjd_start, format='mjd', scale='utc').datetime
                            t_end_fits = Time(mjd_end, format='mjd', scale='utc').datetime
                            print(f'  FITS Header MJD : {t_start_fits} UTC to {t_end_fits} UTC')
                            delta = (pd.Timestamp(t_start_fits, tz='UTC') - start_t).total_seconds()
                            print(f'  Delta (FITS - Filename): {delta} seconds')

print('\n--- PART 2: Actual Subset Extraction Test ---')
hls_df = pd.DataFrame(hls_records)

# Find subset
matched_indices = set()
for _, flare in df_mx.iterrows():
    overlaps = hls_df[(hls_df['start_time'] <= flare['end_time']) & (hls_df['end_time'] >= flare['start_time'])]
    matched_indices.update(overlaps.index.tolist())

matched_df = hls_df.loc[list(matched_indices)]

any_flare_indices = set()
for _, flare in df.iterrows():
    overlaps = hls_df[(hls_df['start_time'] <= flare['end_time']) & (hls_df['end_time'] >= flare['start_time'])]
    any_flare_indices.update(overlaps.index.tolist())

quiet_df = hls_df[~hls_df.index.isin(any_flare_indices)]
sampled_quiet_df = quiet_df.sample(n=len(matched_df), random_state=42)

subset = pd.concat([matched_df, sampled_quiet_df]).drop_duplicates(subset=['filename'])
subset = subset.sort_values('size', ascending=False)

print(f'Subset files: {len(subset)}')
print(f'Subset zip size: {subset["size"].sum() / (1024**3):.2f} GB')

picks = [subset.iloc[0], subset.iloc[len(subset)//4], subset.iloc[len(subset)//2], subset.iloc[-1]]
os.makedirs('d:/hel1os_subset_test', exist_ok=True)

total_zip = 0
total_ext = 0

for i, row in enumerate(picks):
    z = row['filepath']
    dest = f'd:/hel1os_subset_test/test_{i}'
    os.makedirs(dest, exist_ok=True)
    
    with zipfile.ZipFile(z, 'r') as zf:
        zf.extractall(dest)
        
    extracted = sum(os.path.getsize(os.path.join(r, f)) for r, d, files in os.walk(dest) for f in files)
    
    comp = row['size']
    total_zip += comp
    total_ext += extracted
    
    print(f'Extracting: {row["filename"]} ({comp/1024**2:.2f} MB)')
    print(f'  Extracted: {extracted/1024**2:.2f} MB')
    print(f'  Expansion: {extracted/comp:.2f}x')

print('\n--- Subset Results ---')
print(f'Weighted Expansion for Subset: {total_ext/total_zip:.2f}x')
print(f'Projected Extracted Size: {(subset["size"].sum() * (total_ext/total_zip)) / (1024**3):.2f} GB')

shutil.rmtree('d:/tmp_fits', ignore_errors=True)
shutil.rmtree('d:/hel1os_subset_test', ignore_errors=True)
