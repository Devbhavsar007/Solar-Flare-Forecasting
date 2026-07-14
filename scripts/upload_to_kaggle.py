import os
import subprocess
import json
import shutil
import sys

def run_cmd(cmd, cwd=None):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)

def deploy_part(part_num, out_dir, slug, title):
    print(f"\n=============================================")
    print(f"       DEPLOYING PART {part_num}")
    print(f"=============================================\n")
    
    # 1. Build subset
    print(f"1. Copying files to {out_dir}...")
    # Clean directory if it exists to avoid old files
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    run_cmd(f"python scripts/build_curated_subset.py --split-part {part_num} --out-dir {out_dir}")
    
    # 2. Kaggle Init
    print("\n2. Initializing Kaggle dataset metadata...")
    run_cmd(f"python -m kaggle datasets init -p {out_dir}")
    
    # 3. Update Metadata
    print("\n3. Updating dataset-metadata.json...")
    meta_path = os.path.join(out_dir, "dataset-metadata.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
        
    meta['title'] = title
    meta['id'] = meta['id'].replace("INSERT_SLUG_HERE", slug)
    
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
        
    # 4. Upload
    # using `-r zip` since they're already zip, we want Kaggle to extract them, wait, default Kaggle unzips everything. Wait! Kaggle `-r zip` tells the CLI to compress the folder into a zip. We don't want to double compress!
    # Without -r, it just uploads the files. And Kaggle extracts zip files uploaded. So we should NOT use -r zip!
    print("\n4. Uploading to Kaggle...")
    run_cmd(f"python -m kaggle datasets create -p {out_dir}")
    
    print(f"--- Part {part_num} Deployment Complete ---\n")

if __name__ == "__main__":
    try:
        deploy_part(
            part_num=1,
            out_dir=r"d:\solar_flare_subset\hel1os_part1",
            slug="hel1os-flares-curated-part1",
            title="Aditya-L1 HEL1OS Flares Curated Subset (Part 1)"
        )
        
        deploy_part(
            part_num=2,
            out_dir=r"d:\solar_flare_subset\hel1os_part2",
            slug="hel1os-flares-curated-part2",
            title="Aditya-L1 HEL1OS Flares Curated Subset (Part 2)"
        )
        
        print("\n[SUCCESS] Both datasets deployed successfully.")
    except Exception as e:
        print(f"\n[ERROR] Deployment failed: {e}")
        sys.exit(1)
