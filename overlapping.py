import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde as kde
from matplotlib.colors import Normalize
from matplotlib import cm

# Function to calculate densities and colors for a given dataframe
def get_density_colours(df, column_name='size'):
    data = df[column_name].values
    densObj = kde(data)
    
    def make_colours(vals):
        norm = Normalize(vmin=vals.min(), vmax=vals.max())
        colours = [cm.ScalarMappable(norm=norm, cmap='jet').to_rgba(val) for val in vals]
        return colours
    
    densities = densObj(data)
    colours = make_colours(densities)
    
    return data, densities, colours

# Function to plot the data for the first dataframe
def plot_df1(df1, column_name='size'):
    data1, densities1, colours1 = get_density_colours(df1, column_name)

    plt.scatter(range(len(data1)), data1, color=colours1, label=f'{column_name} from df1')
    plt.xlabel('Index')
    plt.ylabel(column_name.capitalize())
    plt.title(f'{column_name.capitalize()} vs Index with Density Colors for df1')
    plt.legend()
    plt.show()

# Function to plot the data for the second dataframe
def plot_df2(df2, column_name='size'):
    data2, densities2, colours2 = get_density_colours(df2, column_name)

    plt.scatter(range(len(data2)), data2, color=colours2, label=f'{column_name} from df2')
    plt.xlabel('Index')
    plt.ylabel(column_name.capitalize())
    plt.title(f'{column_name.capitalize()} vs Index with Density Colors for df2')
    plt.legend()
    plt.show()

# Create two sample dataframes for testing
np.random.seed(0)
df1 = pd.DataFrame({
    'size': np.random.normal(loc=50, scale=10, size=1000)
})

np.random.seed(1)
df2 = pd.DataFrame({
    'size': np.random.normal(loc=50, scale=10, size=1000)
})

# Test the functions
plot_df1(df1)
plot_df2(df2)



--------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde as kde
from matplotlib.colors import Normalize
from matplotlib import cm

# Function to calculate densities and colors for a given dataframe
def get_density_colours(df, column_name='size'):
    data = df[column_name].values
    densObj = kde(data)
    
    def make_colours(vals):
        norm = Normalize(vmin=vals.min(), vmax=vals.max())
        colours = [cm.ScalarMappable(norm=norm, cmap='Blues').to_rgba(val) for val in vals]
        return colours
    
    densities = densObj(data)
    colours = make_colours(densities)
    
    return data, densities, colours

# Function to plot the data for the first dataframe
def plot_df1(df1, column_name='size'):
    data1, densities1, colours1 = get_density_colours(df1, column_name)

    plt.scatter(range(len(data1)), data1, color=colours1, label=f'{column_name} from df1')
    plt.xlabel('Index')
    plt.ylabel(column_name.capitalize())
    plt.title(f'{column_name.capitalize()} vs Index with Density Colors for df1')
    plt.legend()
    plt.show()

# Function to plot the data for the second dataframe
def plot_df2(df2, column_name='size'):
    data2, densities2, colours2 = get_density_colours(df2, column_name)

    plt.scatter(range(len(data2)), data2, color=colours2, label=f'{column_name} from df2')
    plt.xlabel('Index')
    plt.ylabel(column_name.capitalize())
    plt.title(f'{column_name.capitalize()} vs Index with Density Colors for df2')
    plt.legend()
    plt.show()

# Create two sample dataframes for testing
np.random.seed(0)
df1 = pd.DataFrame({
    'size': np.random.normal(loc=50, scale=10, size=1000)
})

np.random.seed(1)
df2 = pd.DataFrame({
    'size': np.random.normal(loc=50, scale=10, size=1000)
})

# Test the functions
plot_df1(df1)
plot_df2(df2)
