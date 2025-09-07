import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
# Import data
df = pd.read_csv('medical_examination.csv')

# 2
# Add 'overweight' column: BMI = kg / (m^2); overweight if BMI > 25
bmi = df['weight'] / ((df['height'] / 100) ** 2)
df['overweight'] = (bmi > 25).astype(int)

# 3
# Normalize cholesterol and gluc: 0 = good, 1 = bad
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# 4
def draw_cat_plot():
    # 5
    # Melt df to long format for chosen categorical variables
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    # 6
    # Group and reformat to show counts per 'cardio', 'variable', and 'value'
    df_cat = (
        df_cat
        .value_counts(['cardio', 'variable', 'value'])
        .reset_index(name='total')
        .sort_values(['cardio', 'variable', 'value'])
    )

    # 7
    # Create categorical plot
    g = sns.catplot(
        data=df_cat,
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        kind='bar',
        height=5,
        aspect=1
    )
    g.set_axis_labels("variable", "total")

    # 8
    fig = g.fig

    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    # Clean the data according to constraints
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ].copy()

    # 12
    # Compute correlation matrix
    corr = df_heat.corr()

    # 13
    # Generate mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14
    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # 15
    # Draw the heatmap
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".1f",
        center=0,
        vmax=0.3,
        vmin=-0.3,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        ax=ax
    )

    # 16
    fig.savefig('heatmap.png')
    return fig