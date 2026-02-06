import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files
df_tree = pd.read_csv('gowalla_tree_extra_level_dist_table_s86400_q130500.csv')
df_3dag = pd.read_csv('gowalla_dag3_extra_level_dist_table_s86400_q130500.csv')
df_5dag = pd.read_csv('gowalla_dag5_extra_level_dist_table_s86400_q130500.csv')

# Convert 'level' to numeric
df_tree['level'] = pd.to_numeric(df_tree['level'], errors='coerce')
df_3dag['level'] = pd.to_numeric(df_3dag['level'], errors='coerce')
df_5dag['level'] = pd.to_numeric(df_5dag['level'], errors='coerce')

# Base levels from 0 to max
max_level = max(df_tree['level'].max(), df_3dag['level'].max(), df_5dag['level'].max())
all_levels = pd.DataFrame({"level": range(0, int(max_level) + 1)})

# Merge with probabilities
df_tree = all_levels.merge(df_tree[['level', 'probability']], on="level", how="left").fillna(0)
df_3dag = all_levels.merge(df_3dag[['level', 'probability']], on="level", how="left").fillna(0)
df_5dag = all_levels.merge(df_5dag[['level', 'probability']], on="level", how="left").fillna(0)

# Plot
bar_width = 0.25
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(all_levels['level'] - bar_width, df_tree['probability'], bar_width, color='green', label='1D-Tree')
ax.bar(all_levels['level'], df_3dag['probability'], bar_width, color='blue', label='3-DAG')
ax.bar(all_levels['level'] + bar_width, df_5dag['probability'], bar_width, color='red', label='5-DAG')
ax.set_xlabel('Returned Level', fontsize=18)
ax.set_ylabel('Probability', fontsize=18)
# ax.set_title('Combined Returned Level Distribution (s=3600, Gowalla)')
ax.set_xticks(all_levels['level'])
ax.tick_params(axis='both', labelsize=12)  # Bigger numbers on axes
ax.legend()
plt.savefig('combined_returned_levels_s86400.png', dpi=300)
plt.show()