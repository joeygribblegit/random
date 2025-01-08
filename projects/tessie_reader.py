import pandas as pd
import geopandas as gpd
import folium
import os
from folium.plugins import MarkerCluster

csv_file_path = '/Users/jgribble/Desktop/5YJ3E1EB0LF711119.csv'

# Load the CSV file and extract only the "Latitude" and "Longitude" columns
columns_to_load = ["Latitude", "Longitude"]
df = pd.read_csv(csv_file_path, usecols=columns_to_load)

# Drop rows with missing values in "Latitude" or "Longitude" columns
df.dropna(subset=["Latitude", "Longitude"], inplace=True)

# Filter out duplicates based on both Latitude and Longitude
df = df.drop_duplicates(subset=['Latitude', 'Longitude'])

# Create a GeoDataFrame by converting Latitude and Longitude into geometry points
gdf = gpd.GeoDataFrame(df, 
                      geometry=gpd.points_from_xy(df.Longitude, df.Latitude))

# Check if there are any rows left in the GeoDataFrame
if not gdf.empty:
    # Create a Folium map centered on the average coordinates
    m = folium.Map(location=[gdf['Latitude'].mean(), gdf['Longitude'].mean()], zoom_start=12)

    # # Plot the car's location data as markers on the map
    # for index, row in gdf.iterrows():
    #     folium.Marker([row['Latitude'], row['Longitude']]).add_to(m)

    # Create a marker cluster
    marker_cluster = MarkerCluster().add_to(m)

    # Add markers to the cluster
    for index, row in gdf.iterrows():
        folium.Marker([row['Latitude'], row['Longitude']]).add_to(marker_cluster)

    # Save the map as an HTML file or display it in a Jupyter Notebook
    # Save the map in the user's home directory
    home_directory = os.path.expanduser("~")
    file_path = os.path.join(home_directory, 'car_location_map.html')
    m.save(file_path)
else:
    print("No valid location data to display on the map.")
