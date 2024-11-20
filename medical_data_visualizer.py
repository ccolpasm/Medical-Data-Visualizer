import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight'] =  (df['weight'] / (df['height'] / 100)**2 > 25).astype(int)

# 3
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)


# 4
def draw_cat_plot():
    # 5
    df_cat = df_cat = pd.melt(df, id_vars=["cardio"], value_vars=["cholesterol", "gluc", "smoke", "alco", "active", "overweight"])
    def map_condition(value):
        if value == 1:
            return "Above normal"
        if value == 2:
            return "Well above normal"
        return "Normal"

    # 6
    #Group and reformat the data in df_cat to split it by cardio
    df_cat['condition'] = df_cat['value'].apply(map_condition)

    # 7
    #Convert the data into long format
    order = ['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke']
    g = sns.catplot(x="variable", hue="condition", col="cardio", data=df_cat, kind="count", order=order)


    # 8
    g.set_axis_labels('variable', 'total')
    fig = g.fig


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &  # La presión diastólica no puede ser mayor que la sistólica
        (df['height'] >= df['height'].quantile(0.025)) &  # Altura mayor que el percentil 2.5
        (df['height'] <= df['height'].quantile(0.975)) &  # Altura menor que el percentil 97.5
        (df['weight'] >= df['weight'].quantile(0.025)) &  # Peso mayor que el percentil 2.5
        (df['weight'] <= df['weight'].quantile(0.975))  # Peso menor que el percentil 97.5
    ]
    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))



    # 14
    fig, ax = plt.subplots(figsize=(12, 10))

    # 15
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", cmap='coolwarm', cbar_kws={'shrink': .8})



    # 16
    fig.savefig('heatmap.png')
    return fig
