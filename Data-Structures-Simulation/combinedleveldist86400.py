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

# Compute CDFs
tree_cdf = df_tree['probability'].cumsum()
dag3_cdf = df_3dag['probability'].cumsum()
dag5_cdf = df_5dag['probability'].cumsum()

# Plot
bar_width = 0.25
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.bar(all_levels['level'] - bar_width, df_tree['probability'], bar_width, color='green', label='1D-Tree')
ax1.bar(all_levels['level'], df_3dag['probability'], bar_width, color='blue', label='3-DAG')
ax1.bar(all_levels['level'] + bar_width, df_5dag['probability'], bar_width, color='red', label='5-DAG')
ax1.set_xlabel('Returned Level', fontsize=20)
ax1.set_ylabel('Probability', fontsize=20)
ax1.set_xticks(all_levels['level'])
ax1.tick_params(axis='both', labelsize=16)
ax1.legend(fontsize=18)  # ← this is kept for bars

# Add second Y-axis for CDF lines
ax2 = ax1.twinx()
ax2.plot(all_levels['level'], tree_cdf, color='green', linestyle='-', label='1D-Tree CDF')
ax2.plot(all_levels['level'], dag3_cdf, color='blue', linestyle='-', label='3-DAG CDF')
ax2.plot(all_levels['level'], dag5_cdf, color='red', linestyle='-', label='5-DAG CDF')
ax2.set_ylabel('Cumulative Probability', fontsize=20)
ax2.set_ylim(0, 1)
ax2.tick_params(axis='y', labelsize=16)

# Combine legends into ONE clean box in upper left
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=18, loc='upper left')

plt.savefig('combined_returned_levels_s86400.png', dpi=300)
plt.show()