# Imports
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns


def show_boxplot(df, column_name, category="is_target", title="", figsize=(14, 5), palette="Set1", order=None, hue=None, describe=True):
    plt.figure(figsize=figsize)  # Figure size

    # Generate boxplot by category
    # Boxplot with outliers
    plt.subplot(1, 2, 1)
    if not category:  # Without Category column
        sns.boxplot(y=column_name, data=df, palette=palette)
    else:
        sns.boxplot(x=category, y=column_name, data=df, palette=palette, order=order, hue=hue)
    plt.title(f"Boxplot - {column_name.upper()}")

    # Boxplot without outliers
    plt.subplot(1, 2, 2)
    if not category:  # Without Category column
        sns.boxplot(y=column_name, data=df, palette=palette, showfliers=False)
    else:
        sns.boxplot(x=category, y=column_name, data=df,
                    palette=palette, order=order, showfliers=False, hue=hue)
    plt.title(f"Boxplot without Outliers - {column_name.upper()}")

    plt.tight_layout()  # Adjust layout
    plt.title(title)  # Title
    plt.show()  # Show plot

    if describe:  # Describe
        if not category:  # Without Category column
            display(df[column_name].describe(percentiles=[.25, .5, .75, 0.9, 0.95]))
        else:
            display(df.groupby(category)[column_name].describe(percentiles=[.25, .5, .75, 0.9, 0.95]))


def show_histogram(df, column_name, category="is_target", title="", figsize=(16, 8), convert_float=False, kde=False,
                   stat="density", orient_h=True, shrink=.6, palette="Set1", describe=True):
    plt.figure(figsize=figsize)  # Figure size

    if convert_float:  # Convert to float to avoid errors
        df[column_name] = df[column_name].astype(float)

    # Histogram
    if orient_h:  # Horizontal
        if not category:  # Without Category column
            sns.histplot(data=df, x=column_name, kde=kde, stat=stat, palette=palette,
                         common_norm=False, shrink=shrink, legend=True)
        else:
            sns.histplot(data=df, x=column_name, hue=category, kde=kde, stat=stat,
                        palette=palette, common_norm=False, multiple="dodge", shrink=shrink, legend=True)
    else:  # Vertical
        if not category:  # Without Category column
            sns.histplot(data=df, y=column_name, kde=kde, stat=stat, palette=palette,
                         common_norm=False, shrink=shrink, legend=True)
        else:
            sns.histplot(data=df, y=column_name, hue=category, kde=kde, stat=stat,
                        palette=palette, common_norm=False, multiple="dodge", shrink=shrink, legend=True)

    plt.title(title)  # Title          
    plt.show()  # Show plot

    if describe:  # Describe
        if not category:  # Without Category column
            display(df[column_name].describe(percentiles=[.25, .5, .75, 0.9, 0.95]))
        else:
            display(df.groupby(category)[column_name].describe(percentiles=[.25, .5, .75, 0.9, 0.95]))


def show_pie_plots(df, column_name, category="is_target", title="", figsize=(12, 6)):
    _, axs = plt.subplots(
        1, len(df[category].unique()), figsize=figsize)  # Generate subplots

    for i, cat in enumerate(df[category].unique()):  # For each category
        axs[i].set_title(f"{column_name} - {cat}")  # Title
        count = df[df[category] == cat][column_name].value_counts()  # Value counts
        axs[i].pie(count, labels=count.index, autopct="%1.1f%%")  # Pie plot

    plt.tight_layout()  # Adjust layout
    plt.title(title)  # Title
    plt.show()  # Show plot


def show_scatterplot(df, x, y, category="is_target", title="", figsize=(16, 8)):
    plt.figure(figsize=figsize)  # Figure size

    # Scatterplot
    if not category:  # Without Category column
        sns.scatterplot(data=df, x=x, y=y)
    else:
        sns.scatterplot(data=df, hue=category, x=x, y=y)
    
    plt.title(title)  # Title
    plt.show()  # Show plot


def plot_correlation_matrix(df, method="spearman", figsize=(16, 16)):
    corr_df = df.corr(method=method)  # Get confusion matrix
    plt.figure(figsize=figsize)  # Figure size
    sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f")
    plt.show()  # Show plot