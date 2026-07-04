import numpy as np
import pandas as pd
from astropy.io import fits
from pathlib import Path

def generate_synthetic_fits():
    """
    Generates minimal valid SoLEXS and HEL1OS FITS files for CI/staging.
    Injects a synthetic M3.0 flare at t=60min in a 120-minute window.
    Outputs to data/synthetic/.
    """
    out_dir = Path("data/synthetic")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 120 minutes of data, 1 second cadence
    n_points = 120 * 60
    times = np.arange(n_points)
    
    # Base background (Poisson noise)
    background = np.random.poisson(lam=10, size=n_points).astype(np.float32)
    
    # Inject synthetic M3.0 flare at t=60min (index 3600)
    flare_peak_idx = 3600
    flare_profile = np.exp(-0.5 * ((times - flare_peak_idx) / 300.0)**2) * 500.0
    
    flux = background + flare_profile
    
    # --- SoLEXS FITS ---
    col1 = fits.Column(name='TIME', format='D', array=times)
    col2 = fits.Column(name='FLUX', format='E', array=flux)
    hdu = fits.BinTableHDU.from_columns([col1, col2])
    
    solexs_path = out_dir / "solexs_synthetic.fits"
    hdu.writeto(solexs_path, overwrite=True)
    print(f"Generated {solexs_path}")
    
    # --- HEL1OS FITS ---
    # Add a slight delay and different noise profile for HEL1OS
    hel1os_flux = background + np.roll(flare_profile, 30) * 0.8
    col_h1 = fits.Column(name='TIME', format='D', array=times)
    col_h2 = fits.Column(name='RATE', format='E', array=hel1os_flux)
    hdu_h = fits.BinTableHDU.from_columns([col_h1, col_h2])
    
    hel1os_path = out_dir / "hel1os_synthetic.fits"
    hdu_h.writeto(hel1os_path, overwrite=True)
    print(f"Generated {hel1os_path}")

if __name__ == "__main__":
    generate_synthetic_fits()
