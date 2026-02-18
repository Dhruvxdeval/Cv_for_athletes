# analysis.py
# Pure data processing layer

import pandas as pd


def analyze_dataframe(df: pd.DataFrame):
    """
    Takes tracking dataframe and returns structured analysis.
    """

    if df.empty:
        return {"error": "Empty dataframe received."}

    # Basic statistics
    stats = df["player_id"].nunique()

    frame_counts = (
        df.groupby("player_id")["frame_id"]
        .count()
        .sort_values(ascending=False)
        .head(10)
        .to_dict()
    )

    return {
        "total_rows": len(df),
        "unique_players": stats,
        "top_players_by_frames": frame_counts
    }
