#!/usr/bin/env python3
"""
plot_viz.py (improved)

Usage (defaults assume your structure):
  python plot_viz.py
  python plot_viz.py --root . --out plots --top-n 25

Changes vs original:
 - More robust loading of loyalty CSV (searches several likely places).
 - If loyalty CSV missing but listening logs + user_favorite_genres exist,
   compute loyalty_score on-the-fly and plot.
 - Extra logging for diagnostics.
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
    plt.figure(figsize=(10, 5))
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
    plt.figure(figsize=(8, 5))
    maxbin = max(15, int(df_night['night_play_count'].max() if not df_night.empty else 0) + 2)
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
    plt.figure(figsize=(10, max(4, top_n * 0.3)))
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
    df = loyalty_df.copy()
    if 'user_id' not in df.columns:
        print("Loyalty file missing user_id column; skipping loyalty plot.")
        return

    if 'loyalty_score' not in df.columns:
        # attempt to infer a candidate column (contains 'loyal' or 'score')
        candidates = [c for c in df.columns if 'loyal' in c.lower() or 'score' in c.lower()]
        if candidates:
            df = df.rename(columns={candidates[0]: 'loyalty_score'})
            print(f"Renamed column '{candidates[0]}' -> 'loyalty_score' for plotting.")
        else:
            print("Loyalty file missing loyalty_score column; skipping loyalty plot.")
            return

    merged = df.merge(night_df[['user_id', 'night_play_proportion']], on='user_id', how='left').fillna(0)
    plt.figure(figsize=(8, 6))
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


def try_find_loyalty_path(output_root):
    """
    Try several likely places for loyalty / genre_loyalty_scores outputs.
    Returns path (dir) or None.
    """
    candidates = [
        os.path.join(output_root, 'genre_loyalty_scores'),
        os.path.join(output_root, 'genre_loyalty_scores_all'),
        os.path.join(output_root, 'genre_loyalty_scores_above_0_8'),
        os.path.join(output_root, 'genre_loyalty_scores_part'),
    ]
    # add any folder under output_root containing 'loyal' or 'genre_loyalty'
    for entry in os.listdir(output_root) if os.path.isdir(output_root) else []:
        if 'loyal' in entry.lower() or 'genre_loyal' in entry.lower():
            candidates.append(os.path.join(output_root, entry))

    # also look through entire tree for any directory or file name that contains 'loyal'
    for p in glob.glob(os.path.join(output_root, '**', '*loyal*.csv'), recursive=True):
        candidates.append(os.path.dirname(p))

    # choose first candidate that exists and has CSVs
    for c in candidates:
        if c and os.path.isdir(c):
            try:
                files = list(glob.glob(os.path.join(c, "*.csv"))) + list(glob.glob(os.path.join(c, "part-*.csv")))
                if files:
                    print(f"Found loyalty candidate: {c} (files: {len(files)})")
                    return c
            except Exception:
                continue
    return None


def compute_loyalty_from_logs_and_user_fav(logs_df, user_fav_dir):
    """
    If loyalty CSV is not available, compute loyalty using:
     - user_fav_dir: Spark output with (user_id, genre, plays) produced by Task 1
     - logs_df: raw listening logs with timestamps
    Returns a dataframe with user_id, top_genre, top_genre_plays, total_plays, loyalty_score
    """
    if logs_df is None:
        print("No raw logs available to compute loyalty.")
        return None
    if not os.path.isdir(user_fav_dir):
        print("user_favorite_genres output not present; cannot compute loyalty from logs.")
        return None

    try:
        user_fav = read_csv_or_dir(user_fav_dir)
    except Exception as e:
        print(f"Failed to load user_favorite_genres from {user_fav_dir}: {e}")
        return None

    # Expect user_fav to have columns: user_id, genre, plays (or similar)
    fav = user_fav.copy()
    # normalize names
    if 'user_id' not in fav.columns:
        possible = [c for c in fav.columns if 'user' in c.lower()]
        if possible:
            fav = fav.rename(columns={possible[0]: 'user_id'})
    if 'plays' not in fav.columns:
        possible = [c for c in fav.columns if 'play' in c.lower() or 'count' in c.lower()]
        if possible:
            fav = fav.rename(columns={possible[0]: 'plays'})

    if 'user_id' not in fav.columns or 'plays' not in fav.columns:
        print("user_favorite_genres file missing expected columns; cannot compute loyalty.")
        return None

    # compute total plays per user from logs_df
    totals = logs_df.groupby('user_id').size().reset_index(name='total_plays')
    merged = fav.merge(totals, on='user_id', how='left').fillna(0)
    merged['top_genre_plays'] = merged['plays'].astype(int)
    merged['total_plays'] = merged['total_plays'].astype(int)
    merged['loyalty_score'] = merged['top_genre_plays'] / merged['total_plays'].replace(0, pd.NA)
    merged['loyalty_score'] = merged['loyalty_score'].fillna(0.0)
    # keep only useful cols
    out = merged[['user_id', 'genre', 'top_genre_plays', 'total_plays', 'loyalty_score']].rename(columns={'genre': 'top_genre'})
    print(f"Computed loyalty for {len(out)} users from raw logs + user_favorite_genres.")
    return out


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
    spark_loyalty = None  # will search
    spark_night = os.path.join(out_dir, 'night_owl_users')

    print("Looking for files:")
    print(f" - listening_logs.csv: {listening_path}")
    print(f" - output dir: {out_dir}")

    logs_df = None
    if os.path.exists(listening_path):
        print("Loading raw listening logs from listening_logs.csv")
        logs_df = read_csv_or_dir(listening_path)
    else:
        print("No listening_logs.csv found in root.")

    # load or compute night_df
    night_df = None
    if logs_df is not None:
        print("Computing night counts from raw logs.")
        night_df, logs_df = compute_night_counts_from_logs(logs_df)
    else:
        # try reading Spark night output (if no raw logs)
        if os.path.isdir(spark_night):
            try:
                print(f"Loading night counts from {spark_night}")
                night_df = read_csv_or_dir(spark_night)
                # normalize column names:
                if 'night_play_count' not in night_df.columns:
                    for c in ['night_play_count', 'night_count', 'night_plays', 'count']:
                        if c in night_df.columns:
                            night_df = night_df.rename(columns={c: 'night_play_count'})
                            break
                if 'user_id' not in night_df.columns:
                    print("Warning: night_owl_users output missing 'user_id'.")
                # compute proportion if total present
                if 'night_play_proportion' not in night_df.columns and 'total_play_count' in night_df.columns:
                    night_df['night_play_proportion'] = night_df['night_play_count'] / night_df['total_play_count'].replace(0, pd.NA)
                    night_df['night_play_proportion'] = night_df['night_play_proportion'].fillna(0.0)
            except Exception as e:
                print("Failed to read night_owl_users:", e)
        else:
            print("No night data available (no logs and no spark night_owl_users).")

    # locate loyalty output (search common names)
    if os.path.isdir(out_dir):
        found = try_find_loyalty_path(out_dir)
        if found:
            spark_loyalty = found

    loyalty_df = None
    if spark_loyalty:
        try:
            print(f"Loading loyalty CSVs from {spark_loyalty}")
            loyalty_df = read_csv_or_dir(spark_loyalty)
        except Exception as e:
            print("Failed to read loyalty csv:", e)
            loyalty_df = None
    else:
        print("No loyalty output directory detected under output/. Will attempt to compute loyalty from logs + user_favorite_genres if possible.")

    # If loyalty_df is still None, try to compute it from logs + user_favorite_genres
    if loyalty_df is None:
        print("Attempting to compute loyalty score from raw logs and user_favorite_genres output...")
        computed = compute_loyalty_from_logs_and_user_fav(logs_df, spark_user_fav)
        if computed is not None:
            loyalty_df = computed
        else:
            print("Could not compute loyalty_df from available inputs.")

    # Now produce plots
    if logs_df is not None:
        plot_hour_histogram(logs_df, outdir)
    else:
        print("Skipping hour histogram (no listening_logs.csv).")

    if night_df is not None:
        # ensure numeric types
        if 'night_play_count' in night_df.columns:
            night_df['night_play_count'] = pd.to_numeric(night_df['night_play_count'], errors='coerce').fillna(0).astype(int)
        else:
            night_df['night_play_count'] = 0
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
        # If loyalty_df has different column names make it compatible
        if 'user_id' in loyalty_df.columns and 'loyalty_score' not in loyalty_df.columns:
            # attempt to find a probable column
            candidates = [c for c in loyalty_df.columns if 'loyal' in c.lower() or 'score' in c.lower()]
            if candidates:
                loyalty_df = loyalty_df.rename(columns={candidates[0]: 'loyalty_score'})
                print(f"Renamed column {candidates[0]} -> loyalty_score in loyalty_df")
        # ensure we have user_id and loyalty_score
        if 'user_id' in loyalty_df.columns and 'loyalty_score' in loyalty_df.columns:
            plot_loyalty_vs_night(loyalty_df, night_df if night_df is not None else pd.DataFrame(columns=['user_id', 'night_play_proportion']), outdir)
        else:
            print("Loyalty data present but missing required columns; skipping loyalty plot.")
    else:
        print("No loyalty/genre file found and could not compute it; skipping loyalty plot.")

    print("Done. Plots saved to:", outdir)


if __name__ == "__main__":
    main()
