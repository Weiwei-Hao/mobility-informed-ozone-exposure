"""Resolve explicit logical inputs against external raw data and bundled data."""
from pathlib import Path
import json

PACKAGE=Path(__file__).resolve().parents[1]
DEFAULT_DATA=PACKAGE/'data'




def resolve_input(logical_path,input_root=None,derived_root=None):
    roots=[]
    for root in [input_root,derived_root or DEFAULT_DATA]:
        if root is None:continue
        root=Path(root).resolve()
        if root not in roots:roots.append(root)
    for root in roots:
        aliases=root/'aliases.json'
        mapping=json.loads(aliases.read_text()) if aliases.is_file() else {}
        path=root/mapping.get(logical_path,logical_path)
        if path.is_file():return path
    return None
