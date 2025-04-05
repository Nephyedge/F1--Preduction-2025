import fastf1
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from matplotlib.patches import Rectangle
warnings.filterwarnings('ignore')

# Setup
cache_dir = 'f1_cache'
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

# 2025 Driver Lineup with actual qualifying positions
drivers_2025 = [
    # Qualifying data from user
    {"position": 1, "driver": "Max Verstappen", "team": "Red Bull Racing", "time": "1:26.983"},
    {"position": 2, "driver": "Lando Norris", "team": "McLaren", "time": "1:26.995"},
    {"position": 3, "driver": "Oscar Piastri", "team": "McLaren", "time": "1:27.027"},
    {"position": 4, "driver": "Charles Leclerc", "team": "Ferrari", "time": "1:27.299"},
    {"position": 5, "driver": "George Russell", "team": "Mercedes", "time": "1:27.318"},
    {"position": 6, "driver": "Andrea Kimi Antonelli", "team": "Mercedes", "time": "1:27.555"},
    {"position": 7, "driver": "Isack Hadjar", "team": "VCARB", "time": "1:27.569"},  # Changed from Racing Bulls to VCARB for consistency
    {"position": 8, "driver": "Lewis Hamilton", "team": "Ferrari", "time": "1:27.610"},
    {"position": 9, "driver": "Alexander Albon", "team": "Williams", "time": "1:27.615"},
    {"position": 10, "driver": "Oliver Bearman", "team": "Haas F1 Team", "time": "1:27.867"},  # Standardized team name
    {"position": 11, "driver": "Pierre Gasly", "team": "Alpine", "time": None},
    {"position": 12, "driver": "Carlos Sainz Jr.", "team": "Williams", "time": None},
    {"position": 13, "driver": "Fernando Alonso", "team": "Aston Martin", "time": None},
    {"position": 14, "driver": "Liam Lawson", "team": "VCARB", "time": None},  # Changed from Racing Bulls to VCARB
    {"position": 15, "driver": "Yuki Tsunoda", "team": "Red Bull Racing", "time": None},
    {"position": 16, "driver": "Nico Hülkenberg", "team": "Kick Sauber", "time": None},  # Standardized team name
    {"position": 17, "driver": "Gabriel Bortoleto", "team": "Kick Sauber", "time": None},
    {"position": 18, "driver": "Esteban Ocon", "team": "Haas F1 Team", "time": None},
    {"position": 19, "driver": "Jack Doohan", "team": "Alpine", "time": None},
    {"position": 20, "driver": "Lance Stroll", "team": "Aston Martin", "time": None},
]

# Create a mapping of driver numbers
driver_mapping = {
    'Max Verstappen': 1,
    'Lando Norris': 4,
    'Oscar Piastri': 81,
    'Charles Leclerc': 16,
    'Lewis Hamilton': 44,
    'George Russell': 63,
    'Andrea Kimi Antonelli': 72,
    'Yuki Tsunoda': 22,
    'Carlos Sainz Jr.': 55,
    'Alexander Albon': 23,
    'Fernando Alonso': 14,
    'Lance Stroll': 18,
    'Pierre Gasly': 10,
    'Jack Doohan': 5,
    'Esteban Ocon': 31,
    'Oliver Bearman': 87,
    'Nico Hülkenberg': 27,
    'Gabriel Bortoleto': 50,
    'Liam Lawson': 30,
    'Isack Hadjar': 41
}

# Add driver numbers to the qualifying data
for driver_data in drivers_2025:
    driver_name = driver_data['driver']
    # Handle variations in driver names
    if driver_name == "Alex Albon":
        driver_name = "Alexander Albon"
    elif driver_name == "Carlos Sainz":
        driver_name = "Carlos Sainz Jr."
    
    driver_data['DriverNumber'] = driver_mapping.get(driver_name, 0)
    driver_data['FullName'] = driver_name

# Convert to DataFrame for easier handling
df_quali = pd.DataFrame(drivers_2025)

# Load historical race data
seasons = [2022, 2023, 2024]
race_data = []
for season in seasons:
    for race in range(1, 23):
        try:
            session = fastf1.get_session(season, race, 'R')
            session.load()
            results = session.results[['DriverNumber', 'Position', 'Points', 'GridPosition']]
            results['Season'] = season
            results['RaceNumber'] = race
            results['Circuit'] = session.event['EventName']
            race_data.append(results)
        except Exception as e:
            print(f"⚠️ Skipping {season} Race {race} — {e}")
            continue

# Ensure there's data
if not race_data:
    raise ValueError("No historical race data was loaded. Check your FastF1 cache or connection.")

# Combine & process
df_all = pd.concat(race_data)
df_all['DriverNumber'] = df_all['DriverNumber'].astype(int)
df_all['Position'] = pd.to_numeric(df_all['Position'], errors='coerce').fillna(25)
df_all['GridPosition'] = pd.to_numeric(df_all['GridPosition'], errors='coerce').fillna(25)
df_all['IsJapanGP'] = df_all['Circuit'].str.contains('Japan|Suzuka', case=False).astype(int)
df_all['ExperienceCount'] = df_all.groupby('DriverNumber').cumcount() + 1

# Add weights for recency - INCREASED weight factor for recent results
df_all['RaceID'] = df_all['Season'] * 100 + df_all['RaceNumber']
max_race_id = df_all['RaceID'].max()
df_all['Recency'] = (df_all['RaceID'] - df_all['RaceID'].min()) / (max_race_id - df_all['RaceID'].min())
df_all['RecencyWeight'] = 1 + 5 * df_all['Recency']  # Recent races weighted up to 6x more

# Rolling average for form
df_all = df_all.sort_values(['DriverNumber', 'Season', 'RaceNumber'])
df_all['RollingPoints'] = (
    df_all.groupby('DriverNumber')['Points']
    .rolling(window=5, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

# Calculate grid-to-finish position delta for each driver
df_all['PositionDelta'] = df_all['GridPosition'] - df_all['Position']
df_all['RollingDelta'] = (
    df_all.groupby('DriverNumber')['PositionDelta']
    .rolling(window=10, min_periods=3)
    .mean()
    .reset_index(level=0, drop=True)
)

# Create recent synthetic race data with focus on driver form
recent_races = [
    # Qatar GP - Latest race with McLaren and Red Bull battle
    {'Race': 'Qatar GP', 'Driver': 'Lando Norris', 'Team': 'McLaren', 'GridPos': 1, 'Position': 1},
    {'Race': 'Qatar GP', 'Driver': 'Oscar Piastri', 'Team': 'McLaren', 'GridPos': 3, 'Position': 2},
    {'Race': 'Qatar GP', 'Driver': 'Max Verstappen', 'Team': 'Red Bull Racing', 'GridPos': 2, 'Position': 3},
    {'Race': 'Qatar GP', 'Driver': 'Charles Leclerc', 'Team': 'Ferrari', 'GridPos': 4, 'Position': 4},
    {'Race': 'Qatar GP', 'Driver': 'Lewis Hamilton', 'Team': 'Ferrari', 'GridPos': 6, 'Position': 5},
    
    # Singapore GP - McLaren strong showing
    {'Race': 'Singapore GP', 'Driver': 'Lando Norris', 'Team': 'McLaren', 'GridPos': 1, 'Position': 1},
    {'Race': 'Singapore GP', 'Driver': 'Charles Leclerc', 'Team': 'Ferrari', 'GridPos': 3, 'Position': 2},
    {'Race': 'Singapore GP', 'Driver': 'Oscar Piastri', 'Team': 'McLaren', 'GridPos': 2, 'Position': 3},
    {'Race': 'Singapore GP', 'Driver': 'Max Verstappen', 'Team': 'Red Bull Racing', 'GridPos': 4, 'Position': 5},
    
    # Australia GP (last round)
    {'Race': 'Australia GP', 'Driver': 'Lando Norris', 'Team': 'McLaren', 'GridPos': 2, 'Position': 1},
    {'Race': 'Australia GP', 'Driver': 'Oscar Piastri', 'Team': 'McLaren', 'GridPos': 3, 'Position': 2},
    {'Race': 'Australia GP', 'Driver': 'Charles Leclerc', 'Team': 'Ferrari', 'GridPos': 5, 'Position': 3},
    {'Race': 'Australia GP', 'Driver': 'Max Verstappen', 'Team': 'Red Bull Racing', 'GridPos': 1, 'Position': 4},
]

# Calculate recent performance metrics for drivers
driver_recent_delta = {}
for race in recent_races:
    driver = race['Driver']
    delta = race['GridPos'] - race['Position']
    if driver in driver_recent_delta:
        driver_recent_delta[driver].append(delta)
    else:
        driver_recent_delta[driver] = [delta]

# Average the deltas
driver_avg_delta = {driver: np.mean(deltas) for driver, deltas in driver_recent_delta.items()}

# Create ML features for prediction
feature_data = []
for driver_data in drivers_2025:
    driver_name = driver_data['FullName']
    team = driver_data['team']
    grid_pos = driver_data['position']
    
    # Get historical race data for this team
    team_hist = df_all[df_all['Team'] == team] if 'Team' in df_all.columns else pd.DataFrame()
    
    # Calculate Experience
    driver_number = driver_data['DriverNumber']
    experience = len(df_all[df_all['DriverNumber'] == driver_number]) if driver_number > 0 else 0
    
    # Get recent performance delta (grid vs finish)
    recent_delta = driver_avg_delta.get(driver_name, 0)
    
    # Assign special factors based on driver and team dynamics
    driver_factor = 1.0
    
    # Adjust based on recent form and qualifying surprise
    if driver_name == "Max Verstappen":
        if grid_pos == 1:  # Max on pole
            driver_factor = 1.15  # Verstappen usually maximizes pole position
    
    elif driver_name == "Lando Norris":
        if grid_pos <= 3:  # Front row or close
            driver_factor = 1.2  # Norris has been getting strong starts
    
    elif driver_name == "Oscar Piastri":
        driver_factor = 1.1  # Piastri has shown good race pace
        
    elif driver_name == "Isack Hadjar":
        if grid_pos == 7:  # Surprising qualifying position
            driver_factor = 0.75  # Might struggle to maintain position
    
    elif driver_name == "Andrea Kimi Antonelli":
        if grid_pos == 6:  # Strong rookie qualifying
            driver_factor = 0.85  # Might lose positions as a rookie
    
    elif driver_name == "Oliver Bearman":
        if grid_pos == 10:  # Top 10 qualifying
            driver_factor = 0.8  # Might struggle with race pace
            
    elif driver_name == "Lewis Hamilton":
        if grid_pos > 5:  # Lower than expected qualifying
            driver_factor = 1.15  # Hamilton tends to recover positions
            
    # Special case for Suzuka-specific skills
    if driver_name in ["Fernando Alonso", "Max Verstappen"]:
        driver_factor *= 1.05  # Historically strong at Suzuka
        
    # Add weather factor - Let's assume a dry race but potential rain midway
    rain_specialist = driver_name in ["Lewis Hamilton", "Max Verstappen", "Fernando Alonso"]
    rain_factor = 1.05 if rain_specialist else 0.95
        
    # Calculate predicted position using qualifying, history and factors
    base_prediction = grid_pos - (recent_delta * 0.7)  # Use 70% of the typical grid-to-finish delta
    
    # Apply specific adjustments
    adjusted_prediction = base_prediction * (1 / driver_factor)
    
    # Add noise/uncertainty (more for midfield, less for front runners)
    uncertainty = 0.2 if grid_pos <= 3 else (0.4 if grid_pos <= 10 else 0.6)
    final_prediction = adjusted_prediction
    
    # Create feature row
    feature_data.append({
        'DriverNumber': driver_number,
        'FullName': driver_name,
        'Team': team,
        'GridPosition': grid_pos,
        'Experience': experience,
        'RecentDelta': recent_delta,
        'PredictedPosition': final_prediction,
        'Uncertainty': uncertainty,
        'DriverFactor': driver_factor,
        'RainFactor': rain_factor
    })

# Create DataFrame with all features
df_prediction = pd.DataFrame(feature_data)

# Run simulation for race outcome
sim_count = 1000
all_results = []

for sim in range(sim_count):
    sim_results = df_prediction.copy()
    
    # Add random noise based on uncertainty
    sim_results['SimPosition'] = sim_results.apply(
        lambda x: max(1, x['PredictedPosition'] + np.random.normal(0, x['Uncertainty'] * 2)), 
        axis=1
    )
    
    # Factor in race incidents - about 10% chance of DNF or significant problem
    incident_mask = np.random.random(len(sim_results)) < 0.1
    sim_results.loc[incident_mask, 'SimPosition'] += np.random.randint(5, 15, size=sum(incident_mask))
    
    # Add first lap chaos factor - front positions are safer
    first_lap_chaos = np.random.normal(0, 3, size=len(sim_results)) * (sim_results['GridPosition'] / 10)
    sim_results['SimPosition'] += first_lap_chaos
    
    # Sort by simulated position for this race
    race_result = sim_results.sort_values('SimPosition').reset_index(drop=True)
    race_result['SimFinish'] = race_result.index + 1
    
    all_results.append(race_result[['DriverNumber', 'FullName', 'SimFinish']])

# Aggregate results from all simulations
final_results = pd.concat(all_results)
avg_positions = final_results.groupby(['DriverNumber', 'FullName'])['SimFinish'].agg(['mean', 'std', 'min', 'max'])
avg_positions = avg_positions.reset_index()

# Join with driver and team info
final_df = pd.merge(avg_positions, df_prediction[['DriverNumber', 'FullName', 'Team', 'GridPosition']], 
                   on=['DriverNumber', 'FullName'])

# Calculate probability of podium finish
podium_counts = {}
for sim_df in all_results:
    for driver in sim_df.loc[sim_df['SimFinish'] <= 3, 'FullName'].unique():
        podium_counts[driver] = podium_counts.get(driver, 0) + 1

# Add podium probability to final dataframe
final_df['PodiumProb'] = final_df['FullName'].map(lambda x: podium_counts.get(x, 0) / sim_count * 100)

# Calculate probability of points finish (top 10)
points_counts = {}
for sim_df in all_results:
    for driver in sim_df.loc[sim_df['SimFinish'] <= 10, 'FullName'].unique():
        points_counts[driver] = points_counts.get(driver, 0) + 1

final_df['PointsProb'] = final_df['FullName'].map(lambda x: points_counts.get(x, 0) / sim_count * 100)

# Sort by predicted average finish position
final_df = final_df.sort_values('mean').reset_index(drop=True)
final_df['PredictedPos'] = final_df.index + 1

# Create team color mapping
team_colors = {
    'McLaren': '#FF8700',          # Orange
    'Red Bull Racing': '#0600EF',  # Dark blue
    'Red Bull': '#0600EF',         # Dark blue
    'Ferrari': '#DC0000',          # Red
    'Mercedes': '#00D2BE',         # Turquoise
    'Aston Martin': '#006F62',     # British racing green
    'Williams': '#005AFF',         # Blue
    'Alpine': '#0090FF',           # Blue
    'Haas F1 Team': '#888888',     # Gray
    'Haas': '#888888',             # Gray
    'Kick Sauber': '#900000',      # Burgundy
    'Sauber': '#900000',           # Burgundy
    'VCARB': '#2B4562',            # Navy blue
    'Racing Bulls': '#2B4562'      # Navy blue
}

# Assign colors to each driver based on team
final_df['Color'] = final_df['Team'].map(team_colors)


# ===== VISUALIZATION 1: Main Race Prediction Chart =====
plt.figure(figsize=(16, 12))

# Plot finishing position bar chart
bars = plt.barh(final_df['FullName'], final_df['mean'], color=final_df['Color'], alpha=0.7)

# Add error bars showing the standard deviation
plt.errorbar(
    final_df['mean'], 
    np.arange(len(final_df)), 
    xerr=final_df['std'],
    fmt='none', 
    color='black', 
    capsize=5,
    alpha=0.5
)

# Add qualifying position markers
plt.scatter(
    final_df['GridPosition'], 
    np.arange(len(final_df)), 
    color='black',
    marker='o', 
    s=100,
    zorder=10,
    label='Qualifying Position'
)

# Add podium probability markers
sizes = final_df['PodiumProb'] * 5 + 20
plt.scatter(
    [1] * len(final_df),  # Fixed x position at 1
    np.arange(len(final_df)),
    s=sizes,
    color=final_df['Color'],
    alpha=0.8,
    edgecolors='black',
    zorder=5
)

# Formatting
plt.xlabel('Predicted Finishing Position (lower is better)', fontsize=14)
plt.title('Suzuka 2025 GP Race Prediction\nBased on Historical Data & Qualifying Results', fontsize=18)
plt.grid(axis='x', linestyle='--', alpha=0.3)

# Add delta markers showing position change from qualifying
for i, (_, row) in enumerate(final_df.iterrows()):
    delta = row['GridPosition'] - row['PredictedPos']
    color = 'green' if delta > 0 else ('red' if delta < 0 else 'gray')
    label = f"+{delta}" if delta > 0 else f"{delta}"
    
    if delta != 0:
        plt.text(
            row['mean'] + row['std'] + 0.3,
            i,
            label,
            va='center',
            fontweight='bold',
            color=color
        )

# Create custom y-tick labels with driver name and team
y_labels = [f"{row['FullName']} ({row['Team']})" for _, row in final_df.iterrows()]
plt.yticks(np.arange(len(final_df)), y_labels)

# Add legend for the qualifying markers
plt.legend(loc='lower right')

# Add explanatory text for podium probability
plt.figtext(0.15, 0.01, 'Circle size at left indicates podium probability', fontsize=12, ha='left')

# Add footnote
plt.figtext(0.5, 0.01, 'Prediction based on qualifying results and historical performance', 
            fontsize=10, ha='center', style='italic')

# Add vertical line for points cutoff (P10)
plt.axvline(x=10.5, color='gray', linestyle='--', alpha=0.5)
plt.text(10.6, len(final_df)-1, 'Points Cutoff', va='center', alpha=0.7)



# Save the figure
plt.tight_layout(rect=[0, 0.02, 1, 0.98])
plt.savefig('suzuka_2025_race_prediction.png', dpi=300, bbox_inches='tight')

# ===== VISUALIZATION 2: Podium Probability Chart =====
plt.figure(figsize=(12, 8))

# Get top 10 drivers by podium probability
top_podium = final_df.sort_values('PodiumProb', ascending=False).head(10)

# Create the bar chart for podium probabilities
podium_bars = plt.bar(
    np.arange(len(top_podium)), 
    top_podium['PodiumProb'], 
    color=top_podium['Color'],
    alpha=0.8,
    edgecolor='black'
)

# Add driver numbers on top of bars
for i, (_, row) in enumerate(top_podium.iterrows()):
    plt.text(
        i, 
        row['PodiumProb'] + 1, 
        f"#{row['DriverNumber']}", 
        ha='center', 
        fontweight='bold'
    )

# Add data labels
for i, bar in enumerate(podium_bars):
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2.,
        height + 0.5,
        f"{height:.1f}%",
        ha='center',
        va='bottom',
        fontsize=10
    )

# Customize the plot
plt.title('Top 10 Drivers - Podium Probability (%)', fontsize=16)
plt.ylabel('Probability (%)', fontsize=12)
plt.ylim(0, top_podium['PodiumProb'].max() + 15)
plt.xticks(np.arange(len(top_podium)), [row['FullName'].split()[0] for _, row in top_podium.iterrows()])
plt.grid(axis='y', linestyle='--', alpha=0.3)


# Save the figure
plt.tight_layout()
plt.savefig('suzuka_2025_podium_probability.png', dpi=300)

# ===== VISUALIZATION 3: Grid vs Predicted Position Comparison =====
plt.figure(figsize=(14, 10))

# Extract data for the plot
grid_pos = final_df['GridPosition']
pred_pos = final_df['PredictedPos']
driver_colors = final_df['Color']
driver_names = [name.split()[0] for name in final_df['FullName']]

# Create scatter plot
plt.scatter(grid_pos, pred_pos, c=driver_colors, s=200, alpha=0.8, edgecolors='black')

# Add driver labels
for i, name in enumerate(driver_names):
    plt.annotate(
        name,
        (grid_pos[i], pred_pos[i]),
        xytext=(5, 5),
        textcoords='offset points',
        fontsize=9,
        fontweight='bold'
    )

# Add diagonal line for reference (no position change)
max_pos = max(grid_pos.max(), pred_pos.max()) + 1
plt.plot([0, max_pos], [0, max_pos], 'k--', alpha=0.3)

# Create zones for gainers and losers
plt.fill_between([0, max_pos], [0, max_pos], [0, 0], alpha=0.1, color='green', label='Position Gained')
plt.fill_between([0, max_pos], [max_pos, max_pos], [0, max_pos], alpha=0.1, color='red', label='Position Lost')

# Customize the plot
plt.title('Grid Position vs Predicted Race Finish', fontsize=16)
plt.xlabel('Grid Position', fontsize=14)
plt.ylabel('Predicted Race Finish', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(loc='lower right')

# Set equal axis ranges
plt.xlim(0.5, max_pos)
plt.ylim(0.5, max_pos)

# Invert both axes (lower position numbers are better)
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()


# Save the figure
plt.tight_layout()
plt.savefig('suzuka_2025_grid_vs_prediction.png', dpi=300)

# ===== VISUALIZATION 4: Points Probability Chart (for Instagram/TikTok) =====
plt.figure(figsize=(10, 10), facecolor='#F5F5F5')

# Create a more visually appealing, square chart for points probability
top_points = final_df.sort_values('PointsProb', ascending=False).head(10)

# Create the horizontal bar chart
points_bars = plt.barh(
    np.arange(len(top_points)), 
    top_points['PointsProb'], 
    color=top_points['Color'],
    alpha=0.8,
    edgecolor='black',
    height=0.6
)

# Add team logo colors as small squares next to driver names
for i, (_, row) in enumerate(top_points.iterrows()):
    plt.plot(0, i, 'o', markersize=15, color=row['Color'], markeredgecolor='black')

# Add data labels inside bars
for i, bar in enumerate(points_bars):
    width = bar.get_width()
    if width > 30:  # Only add text if bar is wide enough
        plt.text(
            width/2,
            bar.get_y() + bar.get_height()/2,
            f"{width:.1f}%",
            ha='center',
            va='center',
            fontsize=12,
            fontweight='bold',
            color='white'
        )

# Customize the plot with a modern look
plt.title('Points Finish Probability - Top 10', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Probability (%)', fontsize=14, labelpad=15)
plt.xlim(0, 105)
plt.xticks(fontsize=12)
plt.yticks(np.arange(len(top_points)), [f"{i+1}. {row['FullName'].split()[0]}" 
                                        for i, (_, row) in enumerate(top_points.iterrows())], 
           fontsize=12, fontweight='bold')
plt.grid(axis='x', linestyle='--', alpha=0.3)

# Add a stylish background for social media appeal
plt.gca().add_patch(Rectangle((0, -0.5), 105, len(top_points), fill=True, 
                              color='#FFFFFF', alpha=0.3, zorder=0))


# Save the figure with square dimensions for Instagram
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig('suzuka_2025_points_probability_social.png', dpi=300)

# ===== VISUALIZATION 5: Driver Head-to-Head Comparison (For Twitter) =====
# Compare top 2 drivers for an engaging Twitter graphic
plt.figure(figsize=(12, 6), facecolor='#F5F5F5')

# Select top 2 drivers
driver1 = final_df.iloc[0]
driver2 = final_df.iloc[1]

# Categories for comparison
categories = ['Grid Pos', 'Pred Pos', 'Podium %', 'Points %', 'Uncertainty']
driver1_values = [driver1['GridPosition'], driver1['PredictedPos'], 
                  driver1['PodiumProb']/100, driver1['PointsProb']/100, 1-driver1['std']/3]
driver2_values = [driver2['GridPosition'], driver2['PredictedPos'], 
                  driver2['PodiumProb']/100, driver2['PointsProb']/100, 1-driver2['std']/3]

# Normalize values for radar chart
def normalize(values):
    normalized = []
    for i, val in enumerate(values):
        if i < 2:  # Positions (lower is better)
            normalized.append(1 - (val - 1) / 19)
        else:  # Percent-based (higher is better)
            normalized.append(val)
    return normalized

# Normalize the values
driver1_norm = normalize(driver1_values)
driver2_norm = normalize(driver2_values)
driver1_norm += [driver1_norm[0]]
driver2_norm += [driver2_norm[0]]

# Set up the radar chart
categories = [*categories, categories[0]]  # Close the loop
driver1_norm = [*driver1_norm, driver1_norm[0]]
driver2_norm = [*driver2_norm, driver2_norm[0]]

# Prepare angles and close the loop
angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
angles += [angles[0]]

# Set up the plot
plt.figure(figsize=(12, 6), facecolor='#F5F5F5')
ax = plt.subplot(111, polar=True)

# Plot the values for both drivers
ax.plot(angles, driver1_norm, 'o-', linewidth=2, label=driver1['FullName'], color=driver1['Color'])
ax.fill(angles, driver1_norm, alpha=0.25, color=driver1['Color'])
ax.plot(angles, driver2_norm, 'o-', linewidth=2, label=driver2['FullName'], color=driver2['Color'])
ax.fill(angles, driver2_norm, alpha=0.25, color=driver2['Color'])

# Set category labels
plt.xticks(angles[:-1], categories, fontsize=12, fontweight='bold')
ax.set_yticklabels([])
ax.set_ylim(0, 1)

# Remove radial labels and set y-limits
ax.set_yticklabels([])
ax.set_ylim(0, 1)

# Add title and legend
plt.title(f'{driver1["FullName"]} vs {driver2["FullName"]} - Suzuka GP', fontsize=16, fontweight='bold', pad=20)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))


# Save the figure
plt.tight_layout()
plt.savefig('suzuka_2025_driver_comparison.png', dpi=300)

# ===== VISUALIZATION 6: Team Performance Overview =====
plt.figure(figsize=(14, 8))

# Group by team and calculate average predicted position
team_perf = final_df.groupby('Team').agg({
    'mean': 'mean',
    'PodiumProb': 'sum',
    'PointsProb': 'mean',
    'Color': 'first'
}).sort_values('mean')

# Create horizontal bar chart for team performance
bars = plt.barh(
    team_perf.index,
    team_perf['mean'],
    color=team_perf['Color'],
    alpha=0.7,
    edgecolor='black',
    height=0.5
)

# Add team podium probability as text
for i, (team, row) in enumerate(team_perf.iterrows()):
    plt.text(
        row['mean'] + 0.3,
        i,
        f"Podium: {row['PodiumProb']:.1f}%",
        va='center',
        fontsize=10,
        fontweight='bold'
    )

# Customize the plot
plt.title('Team Performance Prediction - Suzuka 2025', fontsize=16)
plt.xlabel('Average Predicted Position (lower is better)', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.3)
plt.gca().invert_xaxis()  # Lower numbers (better positions) to the right

# Save the figure
plt.tight_layout()
plt.savefig('suzuka_2025_team_performance.png', dpi=300)

# ===== VISUALIZATION 7: Prediction Uncertainty Chart =====
plt.figure(figsize=(12, 9))

# Create a scatter plot showing prediction uncertainty
plt.scatter(
    final_df['mean'],
    final_df['std'],
    s=200,
    c=final_df['Color'],
    alpha=0.7,
    edgecolors='black'
)

# Add driver names as annotations
for i, row in final_df.iterrows():
    plt.annotate(
        row['FullName'].split()[0],
        (row['mean'], row['std']),
        xytext=(5, 0),
        textcoords='offset points',
        fontsize=10,
        fontweight='bold'
    )

# Add interpretive regions
plt.axhspan(0, 1, alpha=0.1, color='green', label='High Certainty')
plt.axhspan(1, 2, alpha=0.1, color='yellow', label='Medium Certainty')
plt.axhspan(2, 4, alpha=0.1, color='red', label='Low Certainty')

# Customize the plot
plt.title('Prediction Uncertainty by Driver', fontsize=16)
plt.xlabel('Predicted Position (lower is better)', fontsize=12)
plt.ylabel('Standard Deviation (lower means higher certainty)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right')


# Save the figure
plt.tight_layout()
plt.savefig('suzuka_2025_prediction_uncertainty.png', dpi=300)

# ===== Generate Summary Statistics =====
# Create a summary DataFrame for key insights
summary_stats = {
    'Top Predicted Finisher': final_df.iloc[0]['FullName'],
    'Highest Gainer': final_df.loc[(final_df['GridPosition'] - final_df['PredictedPos']).idxmax()]['FullName'],
    'Most Positions Lost': final_df.loc[(final_df['PredictedPos'] - final_df['GridPosition']).idxmax()]['FullName'],
    'Highest Podium Probability': final_df.loc[final_df['PodiumProb'].idxmax()]['FullName'],
    'Most Uncertain Prediction': final_df.loc[final_df['std'].idxmax()]['FullName'],
    'Most Certain Prediction': final_df.loc[final_df['std'].idxmin()]['FullName'],
    'Top Team': team_perf.index[0]
}

summary_df = pd.DataFrame.from_dict(summary_stats, orient='index', columns=['Value'])

# Print summary
print("\n===== SUZUKA 2025 GRAND PRIX PREDICTION SUMMARY =====")
print(summary_df)
print("\nAnalysis complete! All visualizations saved to disk.")

# Create a comprehensive predictions table for export
export_table = final_df[['PredictedPos', 'FullName', 'Team', 'GridPosition', 'mean', 'std', 'PodiumProb', 'PointsProb']]
export_table.columns = ['Predicted Pos', 'Driver', 'Team', 'Grid Pos', 'Avg Finish', 'Uncertainty', 'Podium %', 'Points %']
export_table.to_csv('suzuka_2025_predictions.csv', index=False)

# Show plot summary
print(f"\nGenerated {7} visualization plots:")
print("1. Main Race Prediction Chart")
print("2. Podium Probability Chart")
print("3. Grid vs Predicted Position Comparison")
print("4. Points Probability Chart (Social Media)")
print("5. Driver Head-to-Head Comparison (Twitter)")
print("6. Team Performance Overview")
print("7. Prediction Uncertainty Chart")

# Close all plots
plt.close('all')