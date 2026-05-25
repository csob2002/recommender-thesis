import argparse
from pathlib import Path

import pandas as pd


def preprocess_lastfm(input_file: str, output_file: str) -> pd.DataFrame:
    df = pd.read_csv(
        input_file,
        sep="\t",
        header=None,
        names=["user_id", "item_id", "play_count"],
    )
    out = df[["user_id", "item_id"]].drop_duplicates().copy()
    out["rating"] = 1.0
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_file, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a LastFM user-item-playcount TSV to an implicit rating CSV.")
    parser.add_argument("--in", dest="input_file", default="inter.txt", help="Input TSV file.")
    parser.add_argument("--out", dest="output_file", default="ratings_lastfm_sample.csv", help="Output CSV file.")
    args = parser.parse_args()

    out = preprocess_lastfm(args.input_file, args.output_file)
    print(f"Saved {args.output_file} | rows={len(out):,} | users={out['user_id'].nunique():,} | items={out['item_id'].nunique():,}")


if __name__ == "__main__":
    main()
