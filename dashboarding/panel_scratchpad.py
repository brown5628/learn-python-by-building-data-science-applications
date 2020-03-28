# %%
import pandas as pd
import pylab as plt 
import param
import panel as pn 
import datetime as dt 
pn.extension()
# %%


def interact_example(a=2, b=3):

    plot = plt.figure() 
    ax = plot.add_subplot(111)

    pd.Series({'a':a, 'b':b}).plot(kind='bar', ax=ax)

    plt.tight_layout()
    plt.close(plot)
    return plot 

# %%
pn.interact(interact_example)

# %%
