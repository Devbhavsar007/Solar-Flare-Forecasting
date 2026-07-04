import urllib.request
import json
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

packages = [
    'torch', 'xgboost', 'timesfm', 'momentfm', 'chronos-forecasting', 
    'dspy-ai', 'graphrag', 'flaml', 'mapie', 'shap', 'langgraph', 
    'peft', 'transformers', 'pandera', 'prometheus-client', 'PyYAML', 
    'pandas', 'numpy', 'astropy', 'sunpy', 'scipy', 'joblib'
]

lines = []
for p in packages:
    try:
        url = f'https://pypi.org/pypi/{p}/json'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            version = data['info']['version']
            lines.append(f'{p}=={version}')
            print(f'Fetched {p}=={version}')
    except Exception as e:
        print(f'Error for {p}: {e}')

with open('requirements.txt', 'w') as f:
    f.write('\n'.join(lines) + '\n')
