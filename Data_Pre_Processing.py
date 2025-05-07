#merge All Raw Files(gameinfo,batting,pitching) file
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def process_baseball_data(batting_file, pitching_file, output_file):
    """
    Process baseball data from two initial files and create a new dataset with advanced metrics.

    Parameters:
    batting_file (str): Path to the batting statistics file
    pitching_file (str): Path to the pitching statistics file
    output_file (str): Path to save the output dataset
    """
    print("Loading data files...")

    # Load the batting data
    batting_df = pd.read_csv(batting_file)

    # Load the pitching data
    pitching_df = pd.read_csv(pitching_file)

    # Convert date format in both datasets
    batting_df['date'] = pd.to_datetime(batting_df['date_x'].astype(str), format='%Y%m%d', errors='coerce')
    pitching_df['date'] = pd.to_datetime(pitching_df['date'].astype(str), format='%Y%m%d', errors='coerce')

    print("Processing batting data...")

    # Group batting data by game and team
    batting_game_team = batting_df.groupby(['gid', 'team']).agg({
        'b_pa': 'sum',
        'b_ab': 'sum',
        'b_r': 'sum',
        'b_h': 'sum',
        'b_d': 'sum',
        'b_t': 'sum',
        'b_hr': 'sum',
        'b_rbi': 'sum',
        'b_sf': 'sum',
        'b_hbp': 'sum',
        'b_w': 'sum',
        'b_iw': 'sum',
        'b_k': 'sum',
        'b_sb': 'sum',
        'b_cs': 'sum',
        'opp': 'first',
        'win': 'first',
        'date': 'first'
    }).reset_index()

    print("Processing pitching data...")

    # Group pitching data by game and team
    pitching_game_team = pitching_df.groupby(['gid', 'team']).agg({
        'p_ipouts': 'sum',
        'p_h': 'sum',
        'p_r': 'sum',
        'p_er': 'sum',
        'p_hr': 'sum',
        'p_w': 'sum',
        'p_k': 'sum',
        'p_bfp': 'sum',
        'date': 'first',
        'opp': 'first'
    }).reset_index()

    # Create the output dataframe structure
    output_data = []

    print("Calculating team-level statistics...")

    # Calculate league-wide offensive stats for context (used in WRC+ calculation)
    league_avg_obp = batting_game_team['b_h'].sum() / batting_game_team['b_ab'].sum()
    league_runs_per_pa = batting_game_team['b_r'].sum() / batting_game_team['b_pa'].sum()

    # Process each game for offensive and defensive stats
    unique_games = batting_game_team['gid'].unique()

    for game_id in unique_games:
        # Get teams for this game
        game_data = batting_game_team[batting_game_team['gid'] == game_id]

        if len(game_data) != 2:  # Skip incomplete games
            continue

        # For each team as the offensive team
        for _, team_data in game_data.iterrows():
            offensive_team = team_data['team']
            defensive_team = team_data['opp']
            game_date = team_data['date']

            # Get all games for this offensive team up to this point
            team_games = batting_game_team[(batting_game_team['team'] == offensive_team) &
                                          (batting_game_team['date'] <= game_date)]

            total_games = len(team_games)

            if total_games == 0:
                continue  # Skip if no games found

            # Basic offensive stats
            total_runs = team_games['b_r'].sum()
            total_rbis = team_games['b_rbi'].sum()

            # Calculate batting average
            avg = team_games['b_h'].sum() / team_games['b_ab'].sum() if team_games['b_ab'].sum() > 0 else 0

            # Calculate on-base percentage
            obp_denominator = team_games['b_ab'].sum() + team_games['b_w'].sum() + team_games['b_hbp'].sum() + team_games['b_sf'].sum()
            obp = (team_games['b_h'].sum() + team_games['b_w'].sum() + team_games['b_hbp'].sum()) / obp_denominator if obp_denominator > 0 else 0

            # Calculate slugging percentage
            slg_denominator = team_games['b_ab'].sum()
            slg = (team_games['b_h'].sum() + team_games['b_d'].sum() + 2*team_games['b_t'].sum() + 3*team_games['b_hr'].sum()) / slg_denominator if slg_denominator > 0 else 0

            # Weighted Runs Created Plus (WRC+) - simplified calculation
            # Linear weights for offensive events
            wOBA_numerator = (0.69 * team_games['b_w'].sum() +
                              0.72 * team_games['b_hbp'].sum() +
                              0.89 * (team_games['b_h'].sum() - team_games['b_d'].sum() - team_games['b_t'].sum() - team_games['b_hr'].sum()) +
                              1.27 * team_games['b_d'].sum() +
                              1.62 * team_games['b_t'].sum() +
                              2.10 * team_games['b_hr'].sum())

            wOBA_denominator = team_games['b_ab'].sum() + team_games['b_w'].sum() + team_games['b_sf'].sum() + team_games['b_hbp'].sum()
            wOBA = wOBA_numerator / wOBA_denominator if wOBA_denominator > 0 else 0

            # Calculate WRC+ (100 is league average)
            wRC_plus = ((wOBA / league_avg_obp) * 100) if league_avg_obp > 0 else 0

            # K percentage and BB percentage
            k_percentage = team_games['b_k'].sum() / team_games['b_pa'].sum() if team_games['b_pa'].sum() > 0 else 0
            bb_percentage = team_games['b_w'].sum() / team_games['b_pa'].sum() if team_games['b_pa'].sum() > 0 else 0

            # Base Running Score - simplified
            bsr = (team_games['b_sb'].sum() * 0.2) - (team_games['b_cs'].sum() * 0.4)

            # Get the pitching stats for the defensive team in this game
            defensive_team_pitching = pitching_game_team[(pitching_game_team['team'] == defensive_team) &
                                                        (pitching_game_team['date'] <= game_date)]

            # Pitching stats calculation
            if len(defensive_team_pitching) > 0:
                innings_pitched = defensive_team_pitching['p_ipouts'].sum() / 3  # Convert outs to innings

                if innings_pitched > 0:
                    # Calculate ERA
                    era = (9 * defensive_team_pitching['p_er'].sum()) / innings_pitched

                    # Calculate K/9, HR/9, BB/9
                    k9 = (9 * defensive_team_pitching['p_k'].sum()) / innings_pitched
                    hr9 = (9 * defensive_team_pitching['p_hr'].sum()) / innings_pitched
                    bb9 = (9 * defensive_team_pitching['p_w'].sum()) / innings_pitched
                else:
                    era = 0
                    k9 = 0
                    hr9 = 0
                    bb9 = 0
            else:
                era = 0
                k9 = 0
                hr9 = 0
                bb9 = 0

            # WAR calculation (simplified approximation)
            # This is a very simplified WAR model and will not be as accurate as professional models
            batting_runs = (wOBA - league_avg_obp) * team_games['b_pa'].sum() * 1.15  # Scaling factor
            replacement_level = 20 * (total_games / 162)  # Prorated replacement level
            war = (batting_runs + bsr + replacement_level) / 10  # Runs per win typically around 10

            # Opposing WAR (simplified)
            opposing_war = 0.1 * (100 - era) / 10 if era > 0 else 0

            # Calculate /5 Players stats (top 5 players by OPS)
            player_stats = batting_df[(batting_df['team'] == offensive_team) &
                                     (batting_df['date'] <= game_date) &
                                     (batting_df['b_pa'] > 0)]

            if len(player_stats) >= 5:
                # Calculate OPS for each player
                player_stats['AVG'] = player_stats['b_h'] / player_stats['b_ab'].replace(0, 1)
                player_stats['OBP'] = (player_stats['b_h'] + player_stats['b_w'] + player_stats['b_hbp']) / \
                                     (player_stats['b_ab'] + player_stats['b_w'] + player_stats['b_hbp'] + player_stats['b_sf'])
                player_stats['SLG'] = (player_stats['b_h'] + player_stats['b_d'] + 2*player_stats['b_t'] + 3*player_stats['b_hr']) / \
                                     player_stats['b_ab'].replace(0, 1)

                # Group by player ID and get average stats
                player_agg = player_stats.groupby('id').agg({
                    'AVG': 'mean',
                    'OBP': 'mean',
                    'SLG': 'mean',
                    'b_pa': 'sum',
                    'b_ab': 'sum',
                    'b_k': 'sum',
                    'b_w': 'sum',
                    'b_h': 'sum',
                    'b_hr': 'sum'
                }).reset_index()

                # Calculate OPS and WRC+ for each player
                player_agg['OPS'] = player_agg['OBP'] + player_agg['SLG']

                # Get top 5 players by OPS
                top5_players = player_agg.sort_values('OPS', ascending=False).head(5)

                # Calculate /5 Players metrics
                avg_5players = top5_players['AVG'].mean()
                obp_5players = top5_players['OBP'].mean()
                slg_5players = top5_players['SLG'].mean()

                # Calculate WAR for top 5 players (simplified)
                war_5players = war * 0.5 / 5  # Assuming top 5 players account for 50% of team WAR

                # Calculate WRC+ for top 5 players
                wrc_5players = wRC_plus * 1.1  # Assuming top 5 players perform 10% better than team average

                # Calculate K% and BB% for top 5 players
                k_percentage_5players = top5_players['b_k'].sum() / top5_players['b_pa'].sum() if top5_players['b_pa'].sum() > 0 else 0
                bb_percentage_5players = top5_players['b_w'].sum() / top5_players['b_pa'].sum() if top5_players['b_pa'].sum() > 0 else 0

                # Calculate opposing metrics for top 5 players
                k9_5players = k9 * 0.9  # Simplified
                bb9_5players = bb9 * 0.9  # Simplified
                era_5players = era * 1.6  # Simplified
                opposing_war_5players = opposing_war * 0.05  # Simplified
            else:
                # Default values if not enough player data
                avg_5players = 0
                obp_5players = 0
                slg_5players = 0
                war_5players = 0
                wrc_5players = 0
                k_percentage_5players = 0
                bb_percentage_5players = 0
                k9_5players = 0
                bb9_5players = 0
                era_5players = 0
                opposing_war_5players = 0

            # Calculate /Week stats
            # Get the start of the week for this game date
            week_start = game_date - timedelta(days=game_date.weekday())
            week_end = week_start + timedelta(days=6)

            # Get all games for this offensive team in this week
            week_games = batting_game_team[(batting_game_team['team'] == offensive_team) &
                                          (batting_game_team['date'] >= week_start) &
                                          (batting_game_team['date'] <= week_end)]

            if len(week_games) > 0:
                # Calculate weekly offensive stats
                avg_week = week_games['b_h'].sum() / week_games['b_ab'].sum() if week_games['b_ab'].sum() > 0 else 0

                obp_denominator_week = week_games['b_ab'].sum() + week_games['b_w'].sum() + week_games['b_hbp'].sum() + week_games['b_sf'].sum()
                obp_week = (week_games['b_h'].sum() + week_games['b_w'].sum() + week_games['b_hbp'].sum()) / obp_denominator_week if obp_denominator_week > 0 else 0

                slg_week = (week_games['b_h'].sum() + week_games['b_d'].sum() + 2*week_games['b_t'].sum() + 3*week_games['b_hr'].sum()) / week_games['b_ab'].sum() if week_games['b_ab'].sum() > 0 else 0

                # Weekly WRC+
                wrc_week = wRC_plus * 0.9  # Simplified approximation

                # Weekly WAR
                war_week = war / (total_games / len(week_games)) if total_games > 0 else 0

                # Weekly K% and BB%
                k_percentage_week = week_games['b_k'].sum() / week_games['b_pa'].sum() if week_games['b_pa'].sum() > 0 else 0
                bb_percentage_week = week_games['b_w'].sum() / week_games['b_pa'].sum() if week_games['b_pa'].sum() > 0 else 0

                # Get pitching stats for this week
                week_pitching = pitching_game_team[(pitching_game_team['team'] == defensive_team) &
                                                 (pitching_game_team['date'] >= week_start) &
                                                 (pitching_game_team['date'] <= week_end)]

                if len(week_pitching) > 0:
                    innings_pitched_week = week_pitching['p_ipouts'].sum() / 3

                    if innings_pitched_week > 0:
                        # Weekly pitching stats
                        k9_week = (9 * week_pitching['p_k'].sum()) / innings_pitched_week
                        bb9_week = (9 * week_pitching['p_w'].sum()) / innings_pitched_week
                        era_week = (9 * week_pitching['p_er'].sum()) / innings_pitched_week
                    else:
                        k9_week = 0
                        bb9_week = 0
                        era_week = 0
                else:
                    k9_week = 0
                    bb9_week = 0
                    era_week = 0

                # Weekly opposing WAR
                opposing_war_week = opposing_war / (total_games / len(week_games)) if total_games > 0 else 0
            else:
                # Default values if no weekly data
                avg_week = 0
                obp_week = 0
                slg_week = 0
                war_week = 0
                wrc_week = 0
                k_percentage_week = 0
                bb_percentage_week = 0
                k9_week = 0
                bb9_week = 0
                era_week = 0
                opposing_war_week = 0

            # Get runs scored in this specific game
            runs_scored = team_data['b_r']

            # Win indicator
            is_win = 1 if team_data['win'] == 1 else 0

            # Format date
            formatted_date = game_date.strftime('%d-%m-%Y')

            # Append to output data
            output_data.append({
                'Date': formatted_date,
                'Offensive Team': offensive_team,
                'Defensive Team': defensive_team,
                'Total Games': total_games,
                'Total Runs': total_runs,
                'RBIs': total_rbis,
                'AVG': round(avg, 3),
                'OBP': round(obp, 3),
                'SLG': round(slg, 3),
                'WRC+': round(wRC_plus),
                'WAR': round(war, 1),
                'K Percentage': round(k_percentage, 3),
                'BB Percentage': round(bb_percentage, 3),
                'BSR': round(bsr, 1),
                'Opposing K/9': round(k9, 2),
                'Opposing HR/9': round(hr9, 2),
                'Opposing BB/9': round(bb9, 2),
                'ERA': round(era, 2),
                'Opposing War': round(opposing_war, 1),
                'AVG/5 Players': round(avg_5players, 4),
                'OBP/5 Players': round(obp_5players, 4),
                'SLG/5 Players': round(slg_5players, 4),
                'WAR/5 Players': round(war_5players, 2),
                'WRC+/5 Players': round(wrc_5players, 1),
                'K Percentage/5 Players': round(k_percentage_5players, 4),
                'BB Percentage/5 Players': round(bb_percentage_5players, 4),
                'Opposing K/9/5 Players': round(k9_5players, 3),
                'Opposing BB/9/5 Players': round(bb9_5players, 3),
                'ERA/5 Players': round(era_5players, 3),
                'Opposing WAR/5 Players': round(opposing_war_5players, 2),
                'AVG/Week': round(avg_week, 3),
                'OBP/Week': round(obp_week, 3),
                'SLG/Week': round(slg_week, 3),
                'WAR/Week': round(war_week, 1),
                'WRC+/Week': round(wrc_week),
                'K Percentage/Week': round(k_percentage_week, 3),
                'BB Percentage/Week': round(bb_percentage_week, 3),
                'Opposing K/9/Week': round(k9_week, 2),
                'Opposing BB/9/Week': round(bb9_week, 2),
                'ERA/Week': round(era_week, 2),
                'Opposing WAR/Week': round(opposing_war_week, 1),
                'Runs Scored': int(runs_scored),
                'Win?': is_win
            })

    print("Creating output dataframe...")
    output_df = pd.DataFrame(output_data)

    # Save the output dataset
    output_df.to_csv(output_file, index=False)
    print(f"Output saved to {output_file}")

    return output_df

if __name__ == "__main__":
    # Example usage
    batting_file = "/content/merged_data/merged_baseball_data_1_2000_2003.csv"  # Replace with your batting file
    pitching_file = "/content/filtered_data/pitcher_stats_2_2000_2003.csv"  # Replace with your pitching file
    output_file = "final_output_file.csv"

    # Process the data
    result_df = process_baseball_data(batting_file, pitching_file, output_file)

    # Display sample of the result
    print("\nSample of the output dataset:")
    print(result_df.head())
