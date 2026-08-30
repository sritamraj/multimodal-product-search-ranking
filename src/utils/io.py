import json
from pathlib import Path
import numpy as np

def save_array(path, array):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)

def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2))

def load_json(path):
    return json.loads(Path(path).read_text())
