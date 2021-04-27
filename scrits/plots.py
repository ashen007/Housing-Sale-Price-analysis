import seaborn as sns
from matplotlib import pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
from plotly.offline import plot, download_plotlyjs


def single_count_plot(dataframe, column, savefig=False, path=None, plotname='temp-plot'):
    plt.figure(figsize=[12, 6], dpi=300)
    sns.countplot(x=column, data=dataframe, palette='viridis')

    if savefig is True and path is not None:
        plt.savefig(plotname, path)

    plt.show()
