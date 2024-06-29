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


-----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde as kde
from matplotlib.colors import Normalize
from matplotlib import cm

# Function to calculate densities and colors for a given dataframe
def get_density_colours(df, column_name='size', cmap='Blues'):
    data = df[column_name].values
    densObj = kde(data)
    
    def make_colours(vals):
        norm = Normalize(vmin=vals.min(), vmax=vals.max())
        colours = [cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba(val) for val in vals]
        return colours
    
    densities = densObj(data)
    colours = make_colours(densities)
    
    return data, densities, colours

# Function to plot the data for the first dataframe
def plot_df1(df1, column_name='size'):
    data1, densities1, colours1 = get_density_colours(df1, column_name, cmap='Blues')

    plt.scatter(range(len(data1)), data1, color=colours1, label=f'{column_name} from df1')
    plt.xlabel('Index')
    plt.ylabel(column_name.capitalize())
    plt.title(f'{column_name.capitalize()} vs Index with Density Colors for df1')
    plt.legend()
    plt.show()

# Function to plot the data for the second dataframe
def plot_df2(df2, column_name='size'):
    data2, densities2, colours2 = get_density_colours(df2, column_name, cmap='Greens')

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

-------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde as kde
from matplotlib.colors import Normalize
from matplotlib import cm

# Function to calculate densities and colors for a given dataframe
def get_density_colours(df, column_name='size', cmap='Blues'):
    data = df[column_name].values
    densObj = kde(data)
    
    def make_colours(vals):
        norm = Normalize(vmin=vals.min(), vmax=vals.max())
        colours = [cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba(val) for val in vals]
        return colours
    
    densities = densObj(data)
    colours = make_colours(densities)
    
    return data, densities, colours

# Function to plot both dataframes together
def plot_combined_density(df1, df2, column_name='size'):
    data1, densities1, colours1 = get_density_colours(df1, column_name, cmap='Blues')
    data2, densities2, colours2 = get_density_colours(df2, column_name, cmap='Greens')
    
    plt.figure(figsize=(10, 6))
    
    # Plot data for df1
    plt.scatter(range(len(data1)), data1, color=colours1, label=f'{column_name} from df1')
    
    # Plot data for df2
    plt.scatter(range(len(data2)), data2, color=colours2, label=f'{column_name} from df2')

    # Identify overlapping points and plot them in yellow
    combined_indices = range(len(data1) + len(data2))
    combined_data = np.concatenate((data1, data2))
    combined_kde = kde(combined_data)
    combined_densities = combined_kde(combined_data)
    combined_norm = Normalize(vmin=combined_densities.min(), vmax=combined_densities.max())
    yellow_colours = [cm.ScalarMappable(norm=combined_norm, cmap='YlOrRd').to_rgba(val) for val in combined_densities]

    plt.scatter(combined_indices, combined_data, color=yellow_colours, label='Overlapping points', alpha=0.6)

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
plot_combined_density(df1, df2)
---------------

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde as kde
from matplotlib.colors import Normalize
from matplotlib import cm

# Function to calculate densities for a given dataframe
def get_density(df, column_name='size'):
    data = df[column_name].values
    densObj = kde(data)
    densities = densObj(data)
    return data, densities

# Function to plot both dataframes together with seaborn
def plot_combined_density(df1, df2, column_name='size'):
    data1, densities1 = get_density(df1, column_name)
    data2, densities2 = get_density(df2, column_name)
    
    # Normalize densities for color mapping
    norm1 = Normalize(vmin=densities1.min(), vmax=densities1.max())
    norm2 = Normalize(vmin=densities2.min(), vmax=densities2.max())
    
    colors1 = [cm.ScalarMappable(norm=norm1, cmap='Blues').to_rgba(val) for val in densities1]
    colors2 = [cm.ScalarMappable(norm=norm2, cmap='Greens').to_rgba(val) for val in densities2]
    
    plt.figure(figsize=(14, 8))
    
    # Scatter plot for df1
    sns.scatterplot(x=range(len(data1)), y=data1, hue=densities1, palette='Blues', legend=False, alpha=0.6)
    
    # Scatter plot for df2
    sns.scatterplot(x=range(len(data2)), y=data2, hue=densities2, palette='Greens', legend=False, alpha=0.6)
    
    # Combine data for overlap
    combined_data = np.concatenate((data1, data2))
    combined_indices = range(len(combined_data))
    combined_kde = kde(combined_data)
    combined_densities = combined_kde(combined_data)
    combined_norm = Normalize(vmin=combined_densities.min(), vmax=combined_densities.max())
    yellow_colours = [cm.ScalarMappable(norm=combined_norm, cmap='YlOrRd').to_rgba(val) for val in combined_densities]
    
    # Plot combined data with yellow color for overlapping points
    plt.scatter(combined_indices, combined_data, color=yellow_colours, label='Overlapping points', alpha=0.3)

    plt.xlabel('Index')
    plt.ylabel(column_name.capitalize())
    plt.title(f'{column_name.capitalize()} vs Index with Density Colors')
    plt.legend()
    plt.show()

# Create two sample dataframes for testing with larger dataset
np.random.seed(0)
df1 = pd.DataFrame({
    'size': np.random.normal(loc=50, scale=10, size=2000000)
})

np.random.seed(1)
df2 = pd.DataFrame({
    'size': np.random.normal(loc=50, scale=10, size=2000000)
})

# Test the function
plot_combined_density(df1, df2)

-------------

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde as kde
from matplotlib.colors import Normalize
from matplotlib import cm

# Function to calculate densities and colors for a given dataframe
def get_density_colours(df, column_name='size', cmap='Blues'):
    data = df[column_name].values
    densObj = kde(data)

    def make_colours(vals):
        norm = Normalize(vmin=vals.min(), vmax=vals.max())
        colours = [cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba(val) for val in vals]
        return colours

    densities = densObj(data)
    colours = make_colours(densities)

    return data, densities, colours

# Function to plot both dataframes together using seaborn
def plot_combined_density(df1, df2, column_name='size'):
    data1, densities1, colours1 = get_density_colours(df1, column_name, cmap='Blues')
    data2, densities2, colours2 = get_density_colours(df2, column_name, cmap='Greens')

    # Combine data and densities
    combined_data = np.concatenate((data1, data2))
    combined_densities = np.concatenate((densities1, densities2))

    # Create a DataFrame to store combined data and densities
    df_combined = pd.DataFrame({
        'Index': np.arange(len(combined_data)),
        'Size': combined_data,
        'Density': combined_densities
    })

    # Plot using seaborn scatterplot with alpha blending for overlapping points
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_combined, x='Index', y='Size', hue='Density', palette='YlOrRd', alpha=0.6)
    
    plt.xlabel('Index')
    plt.ylabel(column_name.capitalize())
    plt.title(f'{column_name.capitalize()} vs Index with Density Colors')
    plt.legend()
    plt.tight_layout()
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
plot_combined_density(df1, df2)

---

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde as kde
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm

# Function to create a custom colormap with lighter shades of blue
def create_light_blue_colormap():
    colors = [(0.9, 0.9, 1.0), (0.6, 0.6, 1.0), (0.3, 0.3, 1.0), (0.0, 0.0, 0.8)]  # Light to dark blue
    cmap = LinearSegmentedColormap.from_list('custom_light_blue', colors)
    return cmap

# Function to calculate densities for a given dataframe
def get_density(df, column_name='size'):
    data = df[column_name].values
    densObj = kde(data)
    densities = densObj(data)
    return data, densities

# Function to plot data from both dataframes separately
def plot_density(df1, df2, column_name='size'):
    data1, densities1 = get_density(df1, column_name)
    data2, densities2 = get_density(df2, column_name)
    
    # Normalize densities for each dataframe
    norm1 = Normalize(vmin=densities1.min(), vmax=densities1.max())
    norm2 = Normalize(vmin=densities2.min(), vmax=densities2.max())
    
    # Generate colors based on normalized densities
    colors1 = [cm.ScalarMappable(norm=norm1, cmap=create_light_blue_colormap()).to_rgba(val) for val in densities1]
    colors2 = [cm.ScalarMappable(norm=norm2, cmap='Greens').to_rgba(val) for val in densities2]
    
    plt.figure(figsize=(14, 8))
    
    # Scatter plot for df1 with custom light blue colormap
    sns.scatterplot(x=range(len(data1)), y=data1, hue=densities1, palette=create_light_blue_colormap(), legend=False, alpha=0.6)
    
    # Scatter plot for df2 with light green to green gradient
    sns.scatterplot(x=range(len(data1), len(data1) + len(data2)), y=data2, hue=densities2, palette='Greens', legend=False, alpha=0.6)

    plt.xlabel('Index')
    plt.ylabel(column_name.capitalize())
    plt.title(f'{column_name.capitalize()} vs Index with Density Colors')
    plt.legend()
    plt.show()

# Create two sample dataframes for testing
np.random.seed(0)
df1 = pd.DataFrame({
    'size': np.random.normal(loc=50, scale=10, size=2000000)
})

np.random.seed(1)
df2 = pd.DataFrame({
    'size': np.random.normal(loc=50, scale=10, size=2000000)
})

# Test the function
plot_density(df1, df2)
