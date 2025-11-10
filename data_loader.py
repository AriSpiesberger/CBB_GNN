import pandas as pd
import numpy as np
import glob
import os
from sklearn.preprocessing import StandardScaler

def preprocess_dataframe(df_in):
    """Helper to pre-process a games dataframe."""
    games_df_out = df_in.dropna(
        subset=['home_score', 'away_score', 'home_short_display_name', 'away_short_display_name', 'date', 'neutral_site']
    ).copy()
    games_df_out['home_score'] = pd.to_numeric(games_df_out['home_score'])
    games_df_out['away_score'] = pd.to_numeric(games_df_out['away_score'])
    games_df_out['date'] = pd.to_datetime(games_df_out['date'])
    games_df_out['full_home_team'] = games_df_out['home_short_display_name']
    games_df_out['full_away_team'] = games_df_out['away_short_display_name']
    games_df_out['mov'] = games_df_out['home_score'] - games_df_out['away_score']
    games_df_out['mov_percentile_rank'] = games_df_out['mov'].rank(pct=True)
    games_df_out['adjacency_value'] = (2 * games_df_out['mov_percentile_rank']) - 1
    return games_df_out

def load_and_preprocess_data(script_dir, season_year):
    """
    Loads all data, preprocesses, normalizes, and creates (Team, Year) nodes.
    Returns a dictionary payload for the trainer.
    """
    
    # --- 2. Load Data ---
    target_file = os.path.join(script_dir, f"mbb_scores_{season_year}.csv")
    print(f"Loading target season data from {target_file}...")
    try:
        df = pd.read_csv(target_file, low_memory=False)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at {target_file}. Please run the R script.")
    
    print(f"Loading historical data...")
    historical_files = glob.glob(os.path.join(script_dir, "mbb_scores_*.csv"))
    historical_files = [f for f in historical_files if f != target_file]
    
    historical_dfs = [pd.read_csv(f, low_memory=False) for f in historical_files]
    historical_games_df = pd.concat(historical_dfs, ignore_index=True)
    print(f"Loaded {len(historical_games_df)} total historical games.")

    # --- 3. Pre-process Data ---
    games_df = preprocess_dataframe(df)
    historical_games_df = preprocess_dataframe(historical_games_df)
    
    all_games_df = pd.concat([games_df, historical_games_df], ignore_index=True)
    print(f"Created master dataset with {len(all_games_df)} total games.")

    # --- 3.6 NEW: Define and Normalize Edge Features ---
    print("Normalizing Edge Features (Box Scores)...")
    TEAM_STATS = ['fgm', 'fga', 'three_pm', 'three_pa', 'ftm', 'fta', 'oreb', 'dreb', 'reb', 'ast', 'stl', 'blk', 'tov', 'pf']
    SRC_COLS = [f"{stat}_home" for stat in TEAM_STATS]
    DST_COLS = [f"{stat}_away" for stat in TEAM_STATS]
    BOX_SCORE_COLS = SRC_COLS + DST_COLS

    all_games_df[BOX_SCORE_COLS] = all_games_df[BOX_SCORE_COLS].fillna(0)
    scaler = StandardScaler()
    scaler.fit(all_games_df[BOX_SCORE_COLS])
    all_games_df[BOX_SCORE_COLS] = scaler.transform(all_games_df[BOX_SCORE_COLS])
    print("Normalization complete.")

    # --- 4. Get Unique Nodes (Team + Year) ---
    print("Creating (Team, Year) nodes...")
    if 'season' not in all_games_df.columns:
        raise ValueError("Error: 'season' column not found in data.")
        
    all_games_df['home_team_year'] = all_games_df['full_home_team'] + "_" + all_games_df['season'].astype(str)
    all_games_df['away_team_year'] = all_games_df['full_away_team'] + "_" + all_games_df['season'].astype(str)
    
    all_nodes = sorted(list(set(np.concatenate([
        all_games_df['home_team_year'].unique(),
        all_games_df['away_team_year'].unique()
    ]))))
    node_to_idx = {node_name: idx for idx, node_name in enumerate(all_nodes)}
    print(f"Found {len(all_nodes)} unique (Team, Year) nodes.")

    # Return all the processed data
    return {
        "all_games_df": all_games_df,
        "all_nodes": all_nodes,
        "node_to_idx": node_to_idx,
        "src_cols": SRC_COLS,
        "dst_cols": DST_COLS
    }