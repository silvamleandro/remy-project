# plots.py

# Imports
import matplotlib.pyplot as plt

def plot_lon_lat(df, long_column_name, lat_column_name):
    '''
        :Param:
            df: DataFrame where data is plotted
            long_column_name: longitude column name
            lat_column_name: latitude column name
            
        :Return:
            Latitude and longitude plot

        :Description:
            Show the latitude and longitude of a flight
    '''

    # Plot longitude and latitude
    plt.scatter(df[long_column_name], y=df[lat_column_name])
    plt.xlabel("Longitude") # x-axis
    plt.ylabel("Latitude") # y-axis
    plt.show() # Plot show
