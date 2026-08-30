import gzip
import pandas as pd
catalog_path = "data/catalog.csv"
images_path = "data/images.csv.gz"
catalog = pd.read_csv(catalog_path)
images = pd.read_csv(gzip.open(images_path))
# The catalog currently stores the image filename in image_path.
# Extract the ABO image_id from the filename.
catalog["image_id"] = (
    catalog["image_path"]
    .fillna("")
    .str.replace(r"^data/images/", "", regex=True)
    .str.replace(r"\.jpg$", "", regex=True)
)
# Map image_id -> actual ABO path
mapping = images[["image_id", "path"]].drop_duplicates("image_id")
catalog = catalog.drop(columns=["image_path"]).merge(
    mapping,
    on="image_id",
    how="left"
)
catalog["image_path"] = catalog["path"].apply(
    lambda x: f"data/images/{x}" if pd.notna(x) else ""
)
catalog = catalog.drop(columns=["path"])
catalog.to_csv(catalog_path, index=False)
print("Catalog updated")
print("Products:", len(catalog))
print("Images mapped:", catalog["image_path"].ne("").sum())
print("Images missing:", catalog["image_path"].eq("").sum())
print(catalog.head().to_string(index=False))
