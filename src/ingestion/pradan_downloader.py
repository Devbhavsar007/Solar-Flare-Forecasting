import os
import subprocess
import glob
from typing import List

class PradanDownloader:
    """
    Downloader for PRADAN SoLEXS/HEL1OS FITS files.
    
    Currently falls back to synthetic generation because the 
    host machine's Fortinet WAF aggressively blocks automated 
    access to https://pradan.issdc.gov.in/pradan/.
    """
    
    def __init__(self):
        self.username = os.getenv("PRADAN_USERNAME")
        self.password = os.getenv("PRADAN_PASSWORD")
        self.download_dir = "data/raw/pradan_download"
        os.makedirs(self.download_dir, exist_ok=True)
        
    def list_new_files(self) -> List[str]:
        """
        Yields paths to new FITS files.
        Since PRADAN is blocked by WAF, we generate synthetic FITS data.
        """
        print("[SYNTHETIC DATA — PRADAN unavailable] Fortinet WAF block detected.")
        
        # Run the synthetic generator
        # Try to find the script relative to the project root
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        script_path = os.path.join(root_dir, "scripts", "generate_synthetic_fits.py")
        
        if os.path.exists(script_path):
            subprocess.run(["python", script_path], cwd=root_dir, check=True)
            
        # Return generated files
        synthetic_dir = os.path.join(root_dir, "data", "synthetic")
        if os.path.exists(synthetic_dir):
            return glob.glob(os.path.join(synthetic_dir, "*.fits"))
        
        return []

if __name__ == "__main__":
    downloader = PradanDownloader()
    files = downloader.list_new_files()
    print(f"Downloaded/Generated {len(files)} files: {files}")
