import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde as kde
from matplotlib.colors import Normalize
from matplotlib import cm

def plot_density_colored_scatter(df1, df2, column_name='size'):
    # Extract the 'size' data
    data1 = df1[column_name].values
    data2 = df2[column_name].values

    # Combine the data for density estimation
    combined_data = np.concatenate([data1, data2])

    # Perform KDE
    densObj = kde(combined_data)

    # Function to create colors based on density
    def makeColours(vals):
        norm = Normalize(vmin=vals.min(), vmax=vals.max())
        colours = [cm.ScalarMappable(norm=norm, cmap='jet').to_rgba(val) for val in vals]
        return colours

    # Get density values for combined data
    densities1 = densObj(data1)
    densities2 = densObj(data2)

    # Get colors for the points based on density
    colours1 = makeColours(densities1)
    colours2 = makeColours(densities2)

    # Plot the data
    plt.scatter(range(len(data1)), data1, color=colours1, label=f'{column_name} from df1')
    plt.scatter(range(len(data2)), data2, color=colours2, label=f'{column_name} from df2')

    plt.xlabel('Index')
    plt.ylabel(column_name.capitalize())
    plt.title(f'{column_name.capitalize()} vs Index with Density Colors')
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

# Test the function
plot_density_colored_scatter(df1, df2)


-------


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
    
    return data, colours

# Function to plot the data
def plot_density_colored_scatter(df1, df2, column_name='size'):
    data1, colours1 = get_density_colours(df1, column_name)
    data2, colours2 = get_density_colours(df2, column_name)

    plt.scatter(range(len(data1)), data1, color=colours1, label=f'{column_name} from df1')
    plt.scatter(range(len(data2)), data2, color=colours2, label=f'{column_name} from df2')

    plt.xlabel('Index')
    plt.ylabel(column_name.capitalize())
    plt.title(f'{column_name.capitalize()} vs Index with Density Colors')
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

# Test the function
plot_density_colored_scatter(df1, df2)

