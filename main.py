# main.py
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, avg, row_number, desc, hour, to_timestamp
)
from pyspark.sql.window import Window

# ===== Configuration (tweak as needed) =====
LISTENING_CSV = "listening_logs.csv"
METADATA_CSV = "songs_metadata.csv"
OUTPUT_DIR = "output"

# Night-listening thresholds (for Task 4)
NIGHT_HOUR_START = 0      # 12 AM inclusive
NIGHT_HOUR_END = 4        # 4:59 AM inclusive (0-4)
NIGHT_PLAY_COUNT_THRESHOLD = 10      # Option A: absolute minimum night plays
NIGHT_PLAY_PROPORTION_THRESHOLD = 0.20  # Option B: fraction of total plays in night window

# ===== Spark session =====
spark = SparkSession.builder.appName("MusicAnalysis").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# ===== Basic I/O checks & make output dirs =====
if not os.path.exists(LISTENING_CSV):
    raise FileNotFoundError(f"Cannot find {LISTENING_CSV} in current directory.")
if not os.path.exists(METADATA_CSV):
    raise FileNotFoundError(f"Cannot find {METADATA_CSV} in current directory.")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== Load datasets =====
logs = spark.read.option("header", "true").option("inferSchema", "true").csv(LISTENING_CSV)
songs = spark.read.option("header", "true").option("inferSchema", "true").csv(METADATA_CSV)

# Parse timestamp column to Spark timestamp type (adjust format if necessary)
logs = logs.withColumn("ts", to_timestamp(col("timestamp"), "yyyy-MM-dd HH:mm:ss"))

# Join logs with song metadata so genre/title/artist are available
logs_with_meta = logs.join(songs, on="song_id", how="left")

# Print dataset counts for sanity
total_logs = logs.count()
total_songs = songs.count()
print(f"Loaded {total_logs} listening logs and {total_songs} songs.")

# ===== Task 1: User Favorite Genres =====
# Count plays per (user_id, genre) and pick genre with highest plays per user
user_genre_counts = logs_with_meta.groupBy("user_id", "genre").agg(count("*").alias("plays"))
w_user = Window.partitionBy("user_id").orderBy(desc("plays"), desc("genre"))  # deterministic tie-breaker
user_fav_genre = user_genre_counts.withColumn("rn", row_number().over(w_user)).filter(col("rn") == 1) \
    .select("user_id", "genre", "plays")

print("\nTask 1: sample user favorite genres (top 10):")
user_fav_genre.show(10, truncate=False)

# Save Task 1 result
user_fav_genre.write.mode("overwrite").option("header", "true") \
    .csv(os.path.join(OUTPUT_DIR, "user_favorite_genres"))

# ===== Task 2: Average Listen Time per Song =====
avg_listen = logs.groupBy("song_id").agg(
    avg(col("duration_sec")).alias("avg_duration_sec"),
    count("*").alias("play_count")
)

avg_listen_with_meta = avg_listen.join(songs, on="song_id", how="left") \
    .select("song_id", "title", "artist", "avg_duration_sec", "play_count") \
    .orderBy(desc("play_count"))

print("\nTask 2: sample average listen time per song (top 10 by play_count):")
avg_listen_with_meta.show(10, truncate=False)

# Save Task 2 result
avg_listen_with_meta.write.mode("overwrite").option("header", "true") \
    .csv(os.path.join(OUTPUT_DIR, "avg_listen_time_per_song"))

# ===== Task 3: Genre Loyalty Scores =====
# total plays per user
user_total_plays = logs_with_meta.groupBy("user_id").agg(count("*").alias("total_plays"))

# top genre plays per user (we already have user_genre_counts)
top_genre_per_user = user_genre_counts.withColumn("rn", row_number().over(w_user)).filter(col("rn") == 1) \
    .select(col("user_id").alias("ug_user"), col("genre").alias("top_genre"), col("plays").alias("top_genre_plays"))

# join to compute loyalty
loyalty = top_genre_per_user.join(user_total_plays, top_genre_per_user.ug_user == user_total_plays.user_id, how="left") \
    .select(
        top_genre_per_user.ug_user.alias("user_id"),
        "top_genre",
        "top_genre_plays",
        "total_plays"
    ) \
    .withColumn("loyalty_score", col("top_genre_plays") / col("total_plays"))

print("\nTask 3: sample genre loyalty scores (top 10 by loyalty):")
loyalty.orderBy(desc("loyalty_score")).show(10, truncate=False)

# Save Task 3 results (all + those above threshold)
loyalty.write.mode("overwrite").option("header", "true").csv(os.path.join(OUTPUT_DIR, "genre_loyalty_scores_all"))
loyalty.filter(col("loyalty_score") > 0.8).write.mode("overwrite").option("header", "true") \
    .csv(os.path.join(OUTPUT_DIR, "genre_loyalty_scores_above_0_8"))

# ===== Task 4: Identify users who listen between 12 AM and 5 AM =====
# extract hour, filter logs in night window
logs_with_hour = logs.withColumn("hour_of_day", hour(col("ts")))

night_logs = logs_with_hour.filter((col("hour_of_day") >= NIGHT_HOUR_START) & (col("hour_of_day") <= NIGHT_HOUR_END))

# Option A: absolute count threshold
night_counts = night_logs.groupBy("user_id").agg(count("*").alias("night_play_count"))
night_users_by_count = night_counts.filter(col("night_play_count") >= NIGHT_PLAY_COUNT_THRESHOLD) \
    .orderBy(desc("night_play_count"))

print("\nTask 4 Option A: users with >= {0} night plays (sample 10):".format(NIGHT_PLAY_COUNT_THRESHOLD))
night_users_by_count.show(10, truncate=False)

# Option B: proportion of plays at night
total_plays_per_user = logs.groupBy("user_id").agg(count("*").alias("total_play_count"))
night_proportions = night_counts.join(total_plays_per_user, on="user_id", how="left") \
    .withColumn("night_play_proportion", col("night_play_count") / col("total_play_count"))

night_users_by_proportion = night_proportions.filter(col("night_play_proportion") >= NIGHT_PLAY_PROPORTION_THRESHOLD) \
    .orderBy(desc("night_play_proportion"))

print("\nTask 4 Option B: users with >= {0:.0%} of plays at night (sample 10):".format(NIGHT_PLAY_PROPORTION_THRESHOLD))
night_users_by_proportion.show(10, truncate=False)

# Save Task 4 results
night_users_by_count.write.mode("overwrite").option("header", "true") \
    .csv(os.path.join(OUTPUT_DIR, "night_owl_users_by_count"))
night_users_by_proportion.write.mode("overwrite").option("header", "true") \
    .csv(os.path.join(OUTPUT_DIR, "night_owl_users_by_proportion"))

# ===== Done =====
print(f"\nAll tasks completed. Results saved under '{OUTPUT_DIR}/'.")
spark.stop()
