import zipfile
import os

with open('top_3_largest.txt', 'r') as f:
    zips = [line.strip() for line in f if line.strip()]

for z in zips:
    if os.path.exists(z):
        try:
            with zipfile.ZipFile(z, 'r') as zf:
                uncompressed = sum(info.file_size for info in zf.infolist())
                comp = os.path.getsize(z)
                ratio = uncompressed / comp if comp > 0 else 0
                name = os.path.basename(z)
                print(f'{name}: {comp/1024**2:.2f} MB -> {uncompressed/1024**2:.2f} MB ({ratio:.2f}x)')
        except Exception as e:
            print(f'Error reading {z}: {e}')
