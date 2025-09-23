#!/usr/bin/env python3
"""
plot_viz.py

Usage (defaults assume your structure):
  python plot_viz.py
  python plot_viz.py --root . --out plots --top-n 25

This script auto-detects:
  listening_logs.csv (root)
  output/user_favorite_genres/
  output/avg_listen_time_per_song/
  output/genre_loyalty_scores/
  output/night_owl_users/

Produces PNGs in the output directory.
"""

import os
import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt

NIGHT_HOURS = set(range(0, 5))  # 0-4 inclusive

def read_csv_or_dir(path, **kwargs):
    """Read either a single CSV file or a directory containing CSVs (Spark part-*.csv)."""
    if path is None:
        return None
    if os.path.isdir(path):
        # gather csv and part-*.csv
        files = sorted(glob.glob(os.path.join(path, "*.csv")))
        if not files:
            files = sorted(glob.glob(os.path.join(path, "part-*.csv")))
        if not files:
            # also accept nested directories like path/part-*/...
            files = sorted(glob.glob(os.path.join(path, "**", "part-*.csv"), recursive=True))
        if not files:
            raise FileNotFoundError(f"No csv files found in directory: {path}")
        dfs = []
        for f in files:
            try:
                dfs.append(pd.read_csv(f, **kwargs))
            except Exception as e:
                print(f"Warning: failed to read {f}: {e}")
        if not dfs:
            raise ValueError(f"No readable CSVs under {path}")
        return pd.concat(dfs, ignore_index=True)
    elif os.path.isfile(path):
        return pd.read_csv(path, **kwargs)
    else:
        return None

def compute_night_counts_from_logs(df):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    if df['timestamp'].isna().any():
        nbad = df['timestamp'].isna().sum()
        print(f"Warning: {nbad} rows have unparsable timestamps and will be dropped.")
    df = df[df['timestamp'].notna()]
    df['hour'] = df['timestamp'].dt.hour
    df['is_night'] = df['hour'].isin(NIGHT_HOURS)
    night_counts = df[df['is_night']].groupby('user_id').size().reset_index(name='night_play_count')
    total_counts = df.groupby('user_id').size().reset_index(name='total_play_count')
    merged = total_counts.merge(night_counts, on='user_id', how='left').fillna(0)
    merged['night_play_count'] = merged['night_play_count'].astype(int)
    merged['night_play_proportion'] = merged['night_play_count'] / merged['total_play_count'].replace(0, pd.NA)
    merged['night_play_proportion'] = merged['night_play_proportion'].fillna(0.0)
    return merged, df

def plot_hour_histogram(df_logs, outdir):
    df = df_logs.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df[df['timestamp'].notna()]
    df['hour'] = df['timestamp'].dt.hour
    counts = df['hour'].value_counts().sort_index()
    all_hours = pd.Series(0, index=range(24))
    counts = all_hours.add(counts, fill_value=0).astype(int)
    plt.figure(figsize=(10,5))
    plt.bar(counts.index, counts.values)
    plt.xlabel('Hour of day (0-23)')
    plt.ylabel('Number of listens')
    plt.title('Listens by hour of day')
    plt.xticks(range(24))
    plt.tight_layout()
    path = os.path.join(outdir, 'hour_of_day_histogram.png')
    plt.savefig(path)
    plt.close()
    print(f"Wrote {path}")

def plot_night_count_histogram(df_night, outdir):
    plt.figure(figsize=(8,5))
    maxbin = max(15, df_night['night_play_count'].max()+2)
    plt.hist(df_night['night_play_count'], bins=range(0, maxbin), edgecolor='black')
    plt.xlabel('Night play count (hours 0-4)')
    plt.ylabel('Number of users')
    plt.title('Distribution of night play counts per user')
    plt.tight_layout()
    path = os.path.join(outdir, 'night_play_count_histogram.png')
    plt.savefig(path)
    plt.close()
    print(f"Wrote {path}")

def plot_top_night_owls(df_night, outdir, top_n=20):
    top = df_night.sort_values('night_play_count', ascending=False).head(top_n)
    if top.empty:
        print("No night owl data to plot for top users.")
        return
    plt.figure(figsize=(10, max(4, top_n*0.3)))
    plt.barh(top['user_id'].astype(str), top['night_play_count'])
    plt.gca().invert_yaxis()
    plt.xlabel('Night play count (0-4h)')
    plt.title(f'Top {top_n} night owl users by night plays')
    plt.tight_layout()
    path = os.path.join(outdir, f'top_{top_n}_night_owls.png')
    plt.savefig(path)
    plt.close()
    print(f"Wrote {path}")

def plot_loyalty_vs_night(loyalty_df, night_df, outdir):
    if loyalty_df is None:
        print("No loyalty dataframe provided; skipping loyalty plot.")
        return
    # normalize column names
    df = loyalty_df.copy()
    if 'user_id' not in df.columns:
        print("Loyalty file missing user_id column; skipping loyalty plot.")
        return
    if 'loyalty_score' not in df.columns:
        # attempt to infer from columns: maybe 'loyalty' or similar
        candidates = [c for c in df.columns if 'loyal' in c.lower() or 'score' in c.lower()]
        if candidates:
            df = df.rename(columns={candidates[0]: 'loyalty_score'})
        else:
            print("Loyalty file missing loyalty_score column; skipping loyalty plot.")
            return
    merged = df.merge(night_df[['user_id', 'night_play_proportion']], on='user_id', how='left').fillna(0)
    plt.figure(figsize=(8,6))
    plt.scatter(merged['loyalty_score'], merged['night_play_proportion'], s=20)
    plt.xlabel('Loyalty score (top genre fraction)')
    plt.ylabel('Night play proportion (0-1)')
    plt.title('Loyalty score vs night play proportion')
    plt.tight_layout()
    path = os.path.join(outdir, 'loyalty_vs_night_scatter.png')
    plt.savefig(path)
    plt.close()
    print(f"Wrote {path}")

def ensure_outdir(path):
    os.makedirs(path, exist_ok=True)
    return path

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='.', help='project root (default ".")')
    p.add_argument('--out', default='plots', help='plots output dir (default "plots")')
    p.add_argument('--top-n', default=20, type=int, help='Top N users for bar chart')
    args = p.parse_args()

    root = os.path.abspath(args.root)
    outdir = ensure_outdir(args.out)

    # expected paths
    listening_path = os.path.join(root, 'listening_logs.csv')
    out_dir = os.path.join(root, 'output')
    spark_user_fav = os.path.join(out_dir, 'user_favorite_genres')
    spark_avg_listen = os.path.join(out_dir, 'avg_listen_time_per_song')
    spark_loyalty = os.path.join(out_dir, 'genre_loyalty_scores')
    spark_night = os.path.join(out_dir, 'night_owl_users')

    print("Looking for files:")
    print(f" - listening_logs.csv: {listening_path}")
    print(f" - output dir: {out_dir} (user_favorite_genres, avg_listen_time_per_song, genre_loyalty_scores, night_owl_users)")

    logs_df = None
    if os.path.exists(listening_path):
        print("Loading raw listening logs from listening_logs.csv")
        logs_df = read_csv_or_dir(listening_path)
    else:
        print("No listening_logs.csv found in root.")

    # Try to read spark night counts if no raw logs
    night_df = None
    if os.path.isdir(spark_night):
        print(f"Loading night counts from {spark_night}")
        try:
            night_df = read_csv_or_dir(spark_night)
            # normalize column names
            if 'night_play_count' not in night_df.columns:
                for c in ['night_play_count', 'night_count', 'night_plays', 'count']:
                    if c in night_df.columns:
                        night_df = night_df.rename(columns={c: 'night_play_count'})
                        break
            if 'user_id' not in night_df.columns:
                # maybe saved as user_id column missing? give warning
                print("Warning: night_owl_users CSV doesn't contain 'user_id' column.")
        except Exception as e:
            print(f"Failed to read night_owl_users: {e}")
            night_df = None

    # Try to read loyalty file (genre_loyalty_scores)
    loyalty_df = None
    if os.path.isdir(spark_loyalty):
        try:
            print(f"Loading loyalty/genre scores from {spark_loyalty}")
            loyalty_df = read_csv_or_dir(spark_loyalty)
        except Exception as e:
            print(f"Failed to read loyalty csv: {e}")
            loyalty_df = None

    # If we have logs, compute night_df (preferred)
    if logs_df is not None:
        print("Computing night counts and hour histogram from raw logs.")
        night_df, logs_df = compute_night_counts_from_logs(logs_df)

    # If we still don't have night_df but have spark_night, try to normalize it (and compute proportion if total provided)
    if night_df is None and os.path.isdir(spark_night):
        print("Attempting to normalize spark night_owl_users output.")
        try:
            # ensure columns
            if 'user_id' in night_df.columns and 'night_play_count' in night_df.columns:
                night_df['night_play_proportion'] = night_df.get('night_play_proportion', 0.0)
            else:
                # attempt to load again robustly
                tmp = read_csv_or_dir(spark_night)
                if 'night_play_count' not in tmp.columns:
                    for c in ['night_play_count', 'night_count', 'night_plays', 'count']:
                        if c in tmp.columns:
                            tmp = tmp.rename(columns={c: 'night_play_count'})
                            break
                if 'total_play_count' in tmp.columns and 'night_play_count' in tmp.columns:
                    tmp['night_play_proportion'] = tmp['night_play_count'] / tmp['total_play_count'].replace(0, pd.NA)
                    tmp['night_play_proportion'] = tmp['night_play_proportion'].fillna(0.0)
                else:
                    tmp['night_play_proportion'] = tmp.get('night_play_proportion', 0.0)
                night_df = tmp
        except Exception as ee:
            print("Could not normalize spark night output:", ee)

    # Now create plots
    if logs_df is not None:
        plot_hour_histogram(logs_df, outdir)
    else:
        print("Skipping hour histogram (no listening_logs.csv).")

    if night_df is not None:
        # ensure integer type
        if 'night_play_count' in night_df.columns:
            night_df['night_play_count'] = pd.to_numeric(night_df['night_play_count'], errors='coerce').fillna(0).astype(int)
        else:
            print("Night dataframe missing night_play_count; creating zero column.")
            night_df['night_play_count'] = 0
        # compute proportion if missing (need total_play_count)
        if 'night_play_proportion' not in night_df.columns:
            if 'total_play_count' in night_df.columns:
                night_df['night_play_proportion'] = night_df['night_play_count'] / night_df['total_play_count'].replace(0, pd.NA)
                night_df['night_play_proportion'] = night_df['night_play_proportion'].fillna(0.0)
            else:
                night_df['night_play_proportion'] = 0.0

        plot_night_count_histogram(night_df, outdir)
        plot_top_night_owls(night_df, outdir, top_n=args.top_n)
    else:
        print("No night counts available; skipping night plots.")

    # loyalty scatter
    if loyalty_df is not None:
        plot_loyalty_vs_night(loyalty_df, night_df if night_df is not None else pd.DataFrame(columns=['user_id','night_play_proportion']), outdir)
    else:
        print("No loyalty/genre file found under output/genre_loyalty_scores; skipping loyalty plot.")

    print("Done. Plots saved to:", outdir)

if __name__ == "__main__":
    main()
