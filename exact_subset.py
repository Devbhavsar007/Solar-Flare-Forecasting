import zipfile
import os
import pandas as pd
import glob
import re

# Load GOES
df = pd.read_parquet('data/raw/noaa_catalog.parquet')
df = df[df['peak_time'] >= pd.Timestamp('2023-09-01', tz='UTC')].copy()
df['start_time'] = pd.to_datetime(df['date'] + df['start'], format='%Y%m%d%H%M', utc=True)
df['end_time'] = pd.to_datetime(df['date'] + df['end'], format='%Y%m%d%H%M', utc=True)
df.loc[df['end_time'] < df['start_time'], 'end_time'] += pd.Timedelta(days=1)
df = df.dropna(subset=['start_time', 'end_time'])
df_mx = df[df['xray_class'].str[0].isin(['M', 'X'])]

zips = glob.glob(r'd:\Users\Nitro 5\Downloads\HEL1OS_Data\**\HLS_*.zip', recursive=True)
pattern = re.compile(r'HLS_(\d{8})_(\d{6})_(\d+)sec_lev1_V\d+')

hls_records = []
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

hls_df = pd.DataFrame(hls_records)

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

total_zip = 0
total_ext = 0

for z in subset['filepath']:
    if os.path.exists(z):
        try:
            with zipfile.ZipFile(z, 'r') as zf:
                uncompressed = sum(info.file_size for info in zf.infolist())
                total_zip += os.path.getsize(z)
                total_ext += uncompressed
        except Exception as e:
            pass

print(f'Exact Uncompressed Size: {total_ext/1024**3:.2f} GB')
print(f'Exact Zipped Size: {total_zip/1024**3:.2f} GB')
if total_zip > 0:
    print(f'Exact True Weighted Expansion: {total_ext/total_zip:.2f}x')

