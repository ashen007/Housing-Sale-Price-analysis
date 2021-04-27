import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
from plotly.offline import plot, download_plotlyjs


def single_count_plot(dataframe, column, savefig=False, path=None, plotname='temp-plot.jpg'):
    plt.figure(figsize=[12, 6], dpi=300)
    sns.countplot(x=column, data=dataframe, palette='viridis')

    if savefig is True and path is not None:
        plt.savefig(plotname, path)

    plt.show()


def line_charts(dataframe, x, y, size=(12, 6), hue=None, style=None, savefig=False, path=None,
                plotname='temp-plot.jpg'):
    plt.figure(figsize=size, dpi=300)
    sns.lineplot(x=x, y=y, data=dataframe, hue=hue, style=style, estimator=None, palette='viridis')
    plt.xticks(rotation=45)

    if savefig is True and path is not None:
        plt.savefig(plotname, path)

    plt.show()


def heat_maps(dataframe, x, y, hue, size=(12, 6), savefig=False, path=None, plotname='temp-plot.jpg'):
    plt.figure(figsize=size, dpi=300)
    temp = pd.pivot_table(data=dataframe, index=x, columns=y, values=hue)
    sns.heatmap(temp)

    if savefig is True and path is not None:
        plt.savefig(plotname, path)

    plt.show()
