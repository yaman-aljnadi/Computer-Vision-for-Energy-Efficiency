# -*- coding: utf-8 -*-
"""
Waskishl
"""

import requests

api_key = "AIzaSyBnuSYPHgNdunNlVVKwhWqMaRzv-Ox7k7k"
url = "https://maps.googleapis.com/maps/api/staticmap?"
zoom = str(18)
scale = str(2)
size = "640x640"
map_type = "satellite"

# Define the initial and ending latitude and longitude values
start_lat = 48.176124
end_lat = 48.140473
start_lng = -94.533720
end_lng = -94.507749

# Counter for first numbering in the file name
counter = 1

# Iterate over the range of latitude values
for lat in range(int(start_lat * 1000000), int(end_lat * 1000000), int(-0.002255 * 1000000)):
    center_lat = lat / 1000000  # Convert back to float
    
    # Counter for second numbering in the file name (reset for each latitude iteration)
    second_counter = 1

    # Iterate over the range of longitude values
    for lng in range(int(start_lng * 1000000), int(end_lng * 1000000), int(0.003433 * 1000000)):
        center_lng = lng / 1000000  # Convert back to float
        center = f"{center_lat},{center_lng}"  # Format center coordinates

        # Make the request for each center
        r = requests.get(url + "center=" + center + "&zoom=" + zoom + "&size=" + size + "&scale=" + scale + "&maptype=" + map_type + "&key=" + api_key + "&sensor=false")

        # Write the response content to files
        file_name = f'C:\\Users\\ayhilal\\Google Drive Streaming\\My Drive\\maps\\Waskishl\\{counter}_{second_counter}_{center_lat},{center_lng}.png'
        with open(file_name, 'wb') as f:
            f.write(r.content)
        
        # Increment second counter for the next longitude iteration
        second_counter += 1
    
    # Increment first counter for the next latitude iteration
    counter += 1