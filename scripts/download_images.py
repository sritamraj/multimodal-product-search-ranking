import subprocess
from pathlib import Path
import pandas as pd
CATALOG = "data/catalog.csv"
BUCKET = "s3://amazon-berkeley-objects/images/small/"
df = pd.read_csv(CATALOG)
paths = (
    df["image_path"]
    .fillna("")
    .str.replace("data/images/", "", regex=False)
)
valid = paths[paths.ne("")]
print(f"Images to download: {len(valid)}")
for n, rel_path in enumerate(valid, 1):
    destination = Path("data/images") / rel_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        continue
    source = BUCKET + rel_path
    result = subprocess.run(
        [
            "aws", "s3", "cp",
            "--no-sign-request",
            source,
            str(destination)
        ],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"[FAILED] {rel_path}")
        print(result.stderr.strip())
    if n % 100 == 0:
        print(f"Progress: {n}/{len(valid)}")
print("Download step finished.")
