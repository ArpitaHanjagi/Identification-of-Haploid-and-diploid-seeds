"""Create a fixed, stratified 70/15/15 train/validation/test split."""
import pandas as pd
from sklearn.model_selection import train_test_split

from common import CLASSES, DATA_DIR, RESULTS_DIR, SPLIT_SEED, SPLITS_CSV

rows = [
    {"path": str(p), "label": idx, "class": name}
    for idx, name in enumerate(CLASSES)
    for p in sorted((DATA_DIR / name).glob("*.jpg"))
]
df = pd.DataFrame(rows)

train, rest = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=SPLIT_SEED)
val, test = train_test_split(rest, test_size=0.50, stratify=rest["label"], random_state=SPLIT_SEED)
df["split"] = "train"
df.loc[val.index, "split"] = "val"
df.loc[test.index, "split"] = "test"

RESULTS_DIR.mkdir(exist_ok=True)
df.to_csv(SPLITS_CSV, index=False)
print(pd.crosstab(df["split"], df["class"], margins=True))
