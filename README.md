# Music Listener Behavior Analysis with Spark Structured API


## Overview
A small, reproducible pipeline that (1) generates synthetic music listening data, (2) runs PySpark analyses to compute per-user / per-song / per-genre metrics, and (3) produces visualizations with pandas + matplotlib.

This README has everything you need: explanation, approach, how to run, expected outputs, examples and troubleshooting.

## Dataset Description
Two CSV files are used:
- **listening_logs.csv**: Contains user listening activity (user_id, song_id, timestamp, duration_sec).
- **songs_metadata.csv**: Contains song details (song_id, title, artist, genre, mood).

## Repository Structure
## Structure of repo
```
├── datagen.py
├── main.py
├── plot_viz.py
├── listening_logs.csv
├── songs_metadata.csv
├── output/
│   ├── user_favorite_genres/
│   ├── avg_listen_time_per_song/
│   ├── genre_loyalty_scores/
│   └── night_owl_users/
└── plots/
    ├── hour_of_day_histogram.png
    ├── night_play_count_histogram.png
    ├── top_20_night_owls.png
    └── loyalty_vs_night_scatter.png
```

## Tasks and Outputs
1. **User's Favourite Genre**: Identifies each user's most listened-to genre. Output: `outputs/user_favorite_genres/`
2. **Average Listen Time Per Song**: Calculates average play duration for each song. Output: `outputs/avg_listen_time_per_song/`
3. **Genre Loyalty Score**: Finds users with a high proportion of plays in their favorite genre. Output: `outputs/genre_loyalty_scores/`
4. **Night Owl Users**: Detects users active between 12 AM and 5 AM. Output: `outputs/night_owl_users/`

## Execution Instructions
1. Install Python and PySpark:
   ```bash
   pip install pyspark
   ```
2. Place `listening_logs.csv` and `songs_metadata.csv` in the project directory.
3. Run the analysis:
   ```bash
   python main.py
   ```
4. Find results in the `outputs/` folder.
  
5. Run the plots by running  `plot_viz.py`
   ```bash
   python plot_viz.py
   ```
## Analysis Workflow
1. **Load Data**: Read both CSV files into Spark DataFrames.
2. **Prepare Data**: Join listening logs with song metadata.
3. **Run Analysis**:
   - Find each user's favorite genre
   - Calculate average listen time per song
   - Compute genre loyalty scores
   - Identify night owl users (12 AM–5 AM)
4. **Save Results**: Each task's output is saved as CSV in the `outputs/` folder.

## Errors and Resolutions
**Night_Owls Data**
- Due to just 1000 logs initially, the night owl count is between 1-5, that is the reason:
  
**Why counts end up around 1–5 (step-by-step)**
   - Total logs = 1000.
   - Total users = 100 → average plays per user = 1000 / 100 = 10.
   -  Night window = hours 0–4 inclusive = 5 hours out of 24 → probability a random listen falls in night window ≈ 5/24 ≈ 0.208.
   -  Expected night plays per user = avg_plays_per_user × p_night ≈ 10 × 0.208 = 2.08.
   -  Treat each user's plays as a Binomial(n = plays_for_user, p = 5/24). For n≈10 and p≈0.208.
   -  variance = n p (1−p) ≈ 10 * 0.208 * 0.792 ≈ 1.65.
   -  standard deviation ≈ √1.65 ≈ 1.28
   -  So most users will have night_play_count within a couple standard deviations of 2. That explains why you see counts like 0,1,2,3,4,5 commonly — 10 or more is rare.

- So I increased the number of logs=5000
- Reason: Increase num_logs (more events): raise num_logs in datagen.py so avg plays/user grows (e.g., 10k logs → avg 100 plays/user → expected night plays ≈ 20).

## Notes
- The analysis uses at least 1000 records in listening logs and 51 in song metadata. (But I used 5000 logs)
- All code and outputs are included in this repository.
- The project structure and output formatting follows the guidelines of the assignment instructions

## Author
Ruthwik Dovala
