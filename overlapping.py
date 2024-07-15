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




def create_light_green_colormap():
    colors = [(0.9, 1.0, 0.9), (0.6, 1.0, 0.6), (0.3, 1.0, 0.3), (0.0, 0.8, 0.0)]  # Light to dark green
    cmap = LinearSegmentedColormap.from_list('custom_light_green', colors)
    return cmap


-----------------

def process_file(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    new_lines = []
    
    for line in lines:
        new_lines.append(line.strip())  # Original line
        words = line.strip().split()
        if words[0].endswith('_a'):
            number = int(''.join(filter(str.isdigit, words[0])))
            half_number = number / 2
            if len(words) > 4:  # Ensure there are at least 5 words
                words[2] = str(float(words[2]) + half_number)
                words[4] = str(float(words[4]) + half_number)
                new_line = ' '.join(words)
                new_lines.append(new_line)

    with open(output_file, 'w') as f:
        for line in new_lines:
            f.write(line + '\n')

# Usage example
input_file = 'input.txt'
output_file = 'output.txt'
process_file(input_file, output_file)


-------------

def process_file(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    new_lines = []
    
    for line in lines:
        new_lines.append(line.strip())  # Original line
        words = line.strip().split()
        if words[0].endswith('_a'):
            number = int(''.join(filter(str.isdigit, words[0])))
            half_number = number / 2
            if len(words) > 4:  # Ensure there are at least 5 words
                words[2] = str(float(words[2]) + half_number)
                words[4] = str(float(words[4]) + half_number)
                words[0] = words[0][:-1] + 'c'  # Change '_a' to '_c' in the first word
                new_line = ' '.join(words)
                new_lines.append(new_line)

    with open(output_file, 'w') as f:
        for line in new_lines:
            f.write(line + '\n')

# Usage example
input_file = 'input.txt'
output_file = 'output.txt'
process_file(input_file, output_file)

------

def process_file(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    new_lines = []

    for line in lines:
        new_lines.append(line.rstrip())  # Original line with its original spacing
        words = line.split()
        if words and words[0].endswith('_a'):
            # Extract number from the first word
            number = int(''.join(filter(str.isdigit, words[0])))
            half_number = number / 2
            new_words = words.copy()
            
            # Modify the third and fifth words
            if len(new_words) > 4:
                new_words[2] = str(float(new_words[2]) + half_number)
                new_words[4] = str(float(new_words[4]) + half_number)
                new_words[0] = new_words[0][:-1] + 'c'  # Change '_a' to '_c' in the first word

            # Construct the new line with the same spacing as the original line
            new_line = line.replace(words[0], new_words[0], 1)
            new_line = new_line.replace(words[2], new_words[2], 1)
            new_line = new_line.replace(words[4], new_words[4], 1)

            new_lines.append(new_line.rstrip())

    with open(output_file, 'w') as f:
        for line in new_lines:
            f.write(line + '\n')

# Usage example
input_file = 'input.txt'
output_file = 'output.txt'
process_file(input_file, output_file)

------

def process_file(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    new_lines = []

    for line in lines:
        new_lines.append(line.rstrip())  # Original line with its original spacing
        words = line.split()
        if words and words[0].endswith('_a'):
            # Extract number from the first word
            number_str = ''.join(filter(str.isdigit, words[0]))
            number = int(number_str) if number_str else 0
            half_number = number / 2.0
            
            if len(words) > 4:
                # Preserve original spacing
                prefix = line.split(words[0])[0]
                space_between_first_and_third = line.split(words[2])[0].split(words[0])[1]
                space_between_third_and_fifth = line.split(words[4])[0].split(words[2])[1]

                new_first_word = words[0][:-1] + 'c'
                new_third_word = str(float(words[2]) + half_number)
                new_fifth_word = str(float(words[4]) + half_number)
                
                new_line = prefix + new_first_word + space_between_first_and_third + new_third_word + space_between_third_and_fifth + new_fifth_word
                new_lines.append(new_line.rstrip())

    with open(output_file, 'w') as f:
        for line in new_lines:
            f.write(line + '\n')

# Usage example
input_file = 'input.txt'
output_file = 'output.txt'
process_file(input_file, output_file)


_____

def process_file(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    new_lines = []

    for line in lines:
        new_lines.append(line.rstrip())  # Add original line
        words = line.split()
        if words and words[0].endswith('_a'):
            # Extract number from the first word
            number_str = ''.join(filter(str.isdigit, words[0]))
            number = int(number_str) if number_str else 0
            half_number = number / 2.0

            # Create a new list of words for the modified line
            new_words = words[:]
            new_words[0] = words[0][:-1] + 'c'
            if len(words) > 4:
                try:
                    new_words[2] = str(float(words[2]) + half_number)
                    new_words[4] = str(float(words[4]) + half_number)
                except ValueError:
                    # In case the 3rd and 5th words are not numbers, skip them
                    pass
            
            # Reconstruct the line while preserving original spacing
            index1 = line.find(words[0])
            index2 = line.find(words[2], index1 + len(words[0]))
            index3 = line.find(words[4], index2 + len(words[2]))
            
            new_line = (line[:index1] + new_words[0] +
                        line[index1 + len(words[0]):index2] + new_words[2] +
                        line[index2 + len(words[2]):index3] + new_words[4] +
                        line[index3 + len(words[4]):])
            
            new_lines.append(new_line.rstrip())

    with open(output_file, 'w') as f:
        for line in new_lines:
            f.write(line + '\n')

# Usage example
input_file = 'input.txt'
output_file = 'output.txt'
process_file(input_file, output_file)

____

def process_file(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    new_lines = []

    for line in lines:
        new_lines.append(line.rstrip())  # Add original line
        words = line.split()
        if words and words[0].endswith('_a'):
            # Extract number from the first word
            number_str = ''.join(filter(str.isdigit, words[0]))
            number = int(number_str) if number_str else 0
            half_number = number / 2.0

            # Create a new list of words for the modified line
            new_words = words[:]
            new_words[0] = words[0][:-1] + 'c'
            if len(words) > 5:
                try:
                    new_words[3] = str(float(words[3]) + half_number)
                    new_words[5] = str(float(words[5]) + half_number)
                except ValueError:
                    # In case the 4th and 6th words are not numbers, skip them
                    pass
            
            # Reconstruct the line while preserving original spacing
            index1 = line.find(words[0])
            index2 = line.find(words[3], index1 + len(words[0]))
            index3 = line.find(words[5], index2 + len(words[3]))
            
            new_line = (line[:index1] + new_words[0] +
                        line[index1 + len(words[0]):index2] + new_words[3] +
                        line[index2 + len(words[3]):index3] + new_words[5] +
                        line[index3 + len(words[5]):])
            
            new_lines.append(new_line.rstrip())

    with open(output_file, 'w') as f:
        for line in new_lines:
            f.write(line + '\n')

# Usage example
input_file = 'input.txt'
output_file = 'output.txt'
process_file(input_file, output_file)

----

def count_lines(file_path):
    with open(file_path, 'r') as file:
        line_count = sum(1 for _ in file)
    return line_count

# Usage
file_path = 'path/to/your/large_file.txt'
number_of_lines = count_lines(file_path)
print(f"The file has {number_of_lines} lines.")


---

#!/bin/bash

# Initialize the total sum variable
total_sum=0

# Iterate over all files in the directory
for file in *; do
  if [[ -f $file ]]; then
    # Sum numbers on lines starting with 'ptablel_after' in the current file
    file_sum=$(grep '^ptablel_after' "$file" | awk '{sum += $2} END {print sum}')
    
    # Add the file sum to the total sum
    total_sum=$((total_sum + file_sum))
  fi
done

# Print the total sum
echo "Total sum: $total_sum"

----

#!/bin/bash

# Initialize the total sum variable
total_sum=0

# Iterate over all files in the directory
for file in *; do
  if [[ -f $file ]]; then
    # Extract lines starting with 'ptablel_after' and sum the numbers in the second column
    file_sum=$(grep '^ptablel_after' "$file" | awk '{sum += $2} END {if (NR > 0) print sum; else print 0}')
    
    # Add the file sum to the total sum
    total_sum=$((total_sum + file_sum))
  fi
done

# Print the total sum
echo "Total sum: $total_sum"

