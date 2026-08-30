import csv
import gzip
import json
from pathlib import Path

INPUT = Path("data/listings.json.gz")
OUTPUT = Path("data/catalog.csv")

MAX_PRODUCTS = 10000


def text_from_value(value):
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, list):
        parts = []

        for item in value:
            if isinstance(item, str):
                parts.append(item)

            elif isinstance(item, dict):
                for key in ("value", "text", "name"):
                    if item.get(key):
                        parts.append(str(item[key]))
                        break

        return " ".join(parts)

    if isinstance(value, dict):
        for key in ("value", "text", "name"):
            if value.get(key):
                return str(value[key])

    return str(value)


count = 0

with gzip.open(INPUT, "rt", encoding="utf-8") as infile, \
     OUTPUT.open("w", newline="", encoding="utf-8") as outfile:

    writer = csv.writer(outfile)

    writer.writerow([
        "product_id",
        "title",
        "description",
        "image_path",
    ])

    for line in infile:
        if count >= MAX_PRODUCTS:
            break

        product = json.loads(line)

        product_id = str(product.get("item_id", "")).strip()
        title = text_from_value(product.get("item_name"))
        description = text_from_value(product.get("bullet_point"))

        image_id = product.get("main_image_id")

        image_path = ""

        if image_id:
            image_path = f"data/images/{image_id}.jpg"

        if product_id and title:
            writer.writerow([
                product_id,
                title,
                description,
                image_path,
            ])

            count += 1

print(f"Created {OUTPUT}")
print(f"Products: {count}")