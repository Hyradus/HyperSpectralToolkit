import os
import re
import numpy as np
import cv2
from bokeh.plotting import figure, curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Select, LinearAxis
from spectral import open_image
from skimage import exposure
from bokeh.models import CustomJS, Span,FixedTicker, RangeTool, Div, BoxAnnotation, PolyDrawTool, ColumnDataSource, PolyDrawTool, Callback, Select, BBoxTileSource, Range1d, CrosshairTool, Span, Slider, Dropdown,TextInput, HoverTool, CustomJSTickFormatter, ColumnDataSource, Button, MultiSelect
from bokeh.models import Select, ColumnDataSource, LinearAxis, Range1d, PolyDrawTool,PointDrawTool,LineEditTool,LabelSet, Label,HTMLTemplateFormatter, PolyEditTool,Slider
from bokeh.models import ColumnDataSource, DataTable, DateFormatter, TableColumn
from bokeh.io import show, export_svgs, export_png
import requests
from zipfile import ZipFile, BadZipFile
import pandas as pd
from selenium import webdriver
from bokeh.io.export import export_svgs
from sklearn.preprocessing import MinMaxScaler

#global lbl_data, img, img_sr, reference_RGB, nan_img_sr_ma, lbl_data, wavelength, reference_RGB, nan_img_sr_ma, plot_height, plot_width

def download_and_extract_specific_folder(url, save_path, extract_dir, target_folder):
    try:
        # Step 1: Download the file from the URL
        response = requests.get(url, timeout=10)  # Timeout added for network issues
        response.raise_for_status()  # Raise HTTPError for bad responses

        # Save the file locally
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"File downloaded successfully and saved as {save_path}.")

        # Step 2: Extract only the specific folder's content
        # Create the directory if it doesn't exist
        if not os.path.exists(extract_dir):
            os.makedirs(extract_dir)

        # Try to extract only the contents of the target folder
        try:
            with ZipFile(save_path, 'r') as zip_ref:
                # Iterate over the files in the ZIP archive
                for file_info in zip_ref.infolist():
                    if file_info.filename.startswith(target_folder):
                        # Extract only files that are in the target folder
                        zip_ref.extract(file_info, extract_dir)
            print(f"Contents of '{target_folder}' extracted successfully into {extract_dir}.")
        except BadZipFile:
            print(f"Error: The downloaded file '{save_path}' is not a valid ZIP file.")
        
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except requests.exceptions.ConnectionError:
        print("Error: Unable to connect to the URL. Check your internet connection or the URL.")
    except requests.exceptions.Timeout:
        print("Error: The request timed out. The server might be slow or unresponsive.")
    except requests.exceptions.RequestException as req_err:
        print(f"An error occurred during the request: {req_err}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Optional: Clean up the downloaded file
        if os.path.exists(save_path):
            os.remove(save_path)  # Remove the downloaded ZIP file after extraction
            print(f"Temporary file '{save_path}' removed.")


home = '/app/'#Path.home()
print(home)
# Directory containing the .hdr and .lbl files
#hdr_folder = '/home/hyradus/SyncThing/SyncData/CRISM'
root_folder = os.path.join(str(home), 'Data')
os.makedirs(f'{home}speclib/', exist_ok=True)                        
spectra_library_path = os.path.join(str(home), 'speclib/mrocr_8001/data')
#spectra_library_path = f'{hdr_folder}/mrocr_8001/data'

# Download and extract Viviano et al 2014 library


url = "https://crismtypespectra.rsl.wustl.edu/mrocr_8001.zip"
save_path = "mrocr_8001.zip"
extract_dir = "/app/speclib"
#extract_dir = f'{hdr_folder}'
target_folder = "mrocr_8001/data/"  # Folder to extract from the ZIP
download_and_extract_specific_folder(url, save_path, extract_dir, target_folder)

from bokeh.models import Range1d

def tickergridder(plot_list, lbl_data, buff=0.5, grid_spacing=0.05):
    """
    Add custom grid lines, ticks, and adjust x/y ranges to a list of plots based on lbl_data.

    Parameters:
    - plot_list: List of Bokeh plot objects to add grids, ticks, and set ranges.
    - lbl_data: Dictionary containing coordinates for longitude and latitude.
    - buff: Buffer around the boundaries to ensure grid coverage.
    - grid_spacing: Spacing between grid lines and ticks.
    """
    # Compute the grid boundaries
    west_lon = round(lbl_data['west_lon'])
    east_lon = round(lbl_data['east_lon'])
    min_lat = round(lbl_data['min_lat'])
    max_lat = round(lbl_data['max_lat'])

    # Create shared Range1d objects for x and y ranges
    x_range = Range1d(start=west_lon, end=east_lon)
    y_range = Range1d(start=min_lat, end=max_lat)

    # Loop over each plot in the list and update ranges and grid lines
    for plot in plot_list:
        # Sync the x_range and y_range of all plots
        #plot.x_range = x_range
        #plot.y_range = y_range

        # Clear existing Span objects (grid lines) to avoid duplication
        plot.renderers = [r for r in plot.renderers if not isinstance(r, Span)]
        
        # Add longitude (x) grid lines
        for x in np.arange(west_lon, east_lon, grid_spacing):
            vline = Span(location=x, dimension='height', line_color='gray', line_dash='dashed', line_width=1, line_alpha=0.9)
            plot.add_layout(vline)

        # Add latitude (y) grid lines
        for y in np.arange(min_lat, max_lat, grid_spacing):
            hline = Span(location=y, dimension='width', line_color='gray', line_dash='dashed', line_width=1, line_alpha=0.9)
            plot.add_layout(hline)

        # Set custom major ticks for x and y axes using FixedTicker
        plot.xaxis[0].ticker = FixedTicker(ticks=np.arange(west_lon, east_lon, grid_spacing))
        plot.yaxis[0].ticker = FixedTicker(ticks=np.arange(min_lat, max_lat, grid_spacing))


# Function to get subdirectories in the root folder
def get_subfolders(directory):
    return [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]

def get_files(directory):    
    # List all hdr files in the folder
    hdr_files = [f for f in os.listdir(directory) if f.endswith('.hdr')]
    lbl_files = [f for f in os.listdir(directory) if f.endswith('.lbl')]
    # Separate the 'if' and 'sr' files
    if_files = [f for f in hdr_files if 'if' in f]
    sr_files = [f for f in hdr_files if 'sr' in f]
    return hdr_files, lbl_files, if_files, sr_files



def update_hdr_files(attr, old, new):
    global dropdown_if, hdr_folder_path
    
    selected_subfolder = subfolder_dropdown.value
    hdr_folder_path = os.path.join(root_folder, selected_subfolder)
    
    # Get new file lists based on the selected folder
    hdr_files, lbl_files, if_files, sr_files = get_files(hdr_folder_path)
    
    # If there are no if_files, handle gracefully (maybe display an empty state)
    if not if_files:
        if_files = ["No I/F files found"]

    # Update or create the dropdown_if widget
    if dropdown_if:
        dropdown_if.options = if_files
        dropdown_if.value = if_files[0]
    else:
        dropdown_if = Select(title="Select I/F File", value=if_files[0], options=if_files)
        dropdown_if.on_change('value', update_files)
    
    # Automatically trigger update_files with the first file
    update_files('value', None, dropdown_if.value)
    
    return hdr_files, lbl_files, if_files, sr_files, dropdown_if, hdr_folder_path, selected_subfolder

# Initialize the subfolder dropdown
subfolder_dropdown = Select(title="Select Subfolder", value=get_subfolders(root_folder)[0], options=get_subfolders(root_folder))
subfolder_dropdown.on_change('value', update_hdr_files)



# Initialize the dropdown menu for 'if' files
selected_subfolder = subfolder_dropdown.value
hdr_folder_path = os.path.join(root_folder, selected_subfolder)
hdr_files, lbl_files, if_files, sr_files = get_files(hdr_folder_path)
print(hdr_folder_path, if_files)
dropdown_if = Select(title="Select I/F File", value=if_files[0], options=if_files)

# ColumnDataSource for band profile
band_profile_source = ColumnDataSource(data=dict(Wavelength=[], Reflectance=[]))



# Band dictionary
band_dict = {
    'MAF': ['OLINDEX3', 'LCPINDEX2', 'HCPINDEX2'],
    'HYD': ['SINDEX2', 'BD2100_2', 'BD1900_2'],
    'PHY': ['D2300', 'D2200', 'BD1900R2'],
    'PFM': ['BD2355', 'D2300', 'BD2290'],
    'PAL': ['BD2210_2', 'BD2190', 'BD2165'],
    'HYS': ['MIN2250', 'BD2250', 'BD1900R2'],
    'ICE': ['BD1900_2', 'BD1500_2', 'BD1435'],
    'IC2': ['R3920', 'BD1500_2', 'BD1435'],
    'CHL': ['ISLOPE1', 'BD3000', 'IRR2'],
    'CAR': ['D2300', 'BD2500_2', 'BD1900_2'],
    'CR2': ['MIN2295_2480', 'MIN2345_2537', 'CINDEX2'],
    'FEM': ['BD530_2', 'SH600_2', 'BDI1000VIS'],
    'FM2': ['BD530_2', 'BD920_2', 'BDI1000VIS'],
    'FALSE': ['R2529','R1506','R1080'],
    'TRUE': ['R600', 'R530', 'R440'],
    'VIS2': ['BD1900_2', 'R1330', 'R770'],
    'TAN': ['R2529', 'R1330', 'R770'],
}

# Available spectral parameters from the current data (replace with actual list)
width = Span(dimension="width")
height = Span(dimension="height")
cht = CrosshairTool(overlay=[width, height])
cht2 = CrosshairTool(overlay=[height,width])


# Function to parse the lbl file and extract lat/lon information
def parse_lbl(lbl_file):
    lbl_data = {}
    with open(lbl_file, 'r') as file:
        for line in file:
            if 'MINIMUM_LATITUDE' in line:
                lbl_data['min_lat'] = float(re.search(r'[-+]?\d*\.\d+|\d+', line).group())
            if 'MAXIMUM_LATITUDE' in line:
                lbl_data['max_lat'] = float(re.search(r'[-+]?\d*\.\d+|\d+', line).group())
            if 'WESTERNMOST_LONGITUDE' in line:
                lbl_data['west_lon'] = float(re.search(r'[-+]?\d*\.\d+|\d+', line).group())
            if 'EASTERNMOST_LONGITUDE' in line:
                lbl_data['east_lon'] = float(re.search(r'[-+]?\d*\.\d+|\d+', line).group())
    return lbl_data


# Function to load the hdr file, its corresponding 'sr' file, and the lbl file

def load_files(if_file):
    #global img, img_sr, wavelength, sr_names, lbl_data, hdr_folder_path
    # Define paths for the selected I/F file, SR file, and LBL file
    selected_subfolder = subfolder_dropdown.value
    hdr_folder_path = os.path.join(root_folder, selected_subfolder)
    hdr_file = os.path.join(hdr_folder_path, if_file)
    sr_file = os.path.join(hdr_folder_path, if_file.replace('if', 'sr'))
    lbl_file = os.path.join(hdr_folder_path, if_file.replace('if', 'in').replace('.hdr', '.lbl'))

    if not os.path.exists(hdr_file) or not os.path.exists(sr_file) or not os.path.exists(lbl_file):
        print(f"Could not find one of the following files: {hdr_file}, {sr_file}, {lbl_file}")
        return None, None, None, None, None

    # Load the files using spectral
    img = open_image(hdr_file)
    img_sr = open_image(sr_file)
    #img_height, img_width, _ = img.shape#[:2]
    print(img.shape)
    # Parse the lbl file to get lat/lon information
    lbl_data = parse_lbl(lbl_file)
    
    # Get metadata
    wavelength = np.array(img.metadata['wavelength']).astype(float)/1000
    sr_names = img_sr.metadata['band names']

    return img, img_sr, wavelength, sr_names, lbl_data, hdr_folder_path#, img_height, img_width


# Function to process and mask the images
def process_image(img, img_sr, channels_names):
    # Mask the 65535 values
    

    # Retrieve the channel indices
    sr_channels_number = [sr_names.index(cn) for cn in channels_names]

    # Prepare RGB browse data
    RGB_browse = np.array(img_sr[:, :, sr_channels_number])
    special_channels = ['R530', 'R440', 'R600', 'R770', 'R1080', 'R1506', 'R2529', 'R3920', 'SH600_2', 'IRA', 'ISLOPE1', 'IRR2']
    p02_val = 50
    p99_val = 90
    # Contrast stretching
    for ind, ch in enumerate(RGB_browse.T):
        if sr_names[sr_channels_number[ind]] in special_channels:
            p02_val = 0.1
            p99_val = 99.9
        ch_mask = ch < 6553
        masked_rgb = ch[ch_mask]
        p99 = np.nanpercentile(masked_rgb, p99_val)
        p02, p98 = np.nanpercentile(masked_rgb, (p02_val, 98))
        print(p02, sr_names[sr_channels_number[ind]], masked_rgb.max(), masked_rgb.min())
        if p99 !=0:
            RGB_browse[:, :, ind] = exposure.rescale_intensity(ch, in_range=(p02, p99)).T
        else:
            RGB_browse[:, :, ind] = 0

    return RGB_browse, nan_img_sr_ma

def plotter(lbl_data, img):
    #global p, p_vis, plot_height, plot_width
    #global plot_height, plot_width,p, p_vis
    #img_height, img_width = img.shape[:2]
    # Calculate aspect ratio based on coordinates
    latitude_range = lbl_data['max_lat'] - lbl_data['min_lat']
    longitude_range = lbl_data['east_lon'] - lbl_data['west_lon']
    aspect_ratio = latitude_range / longitude_range
    print(aspect_ratio)
    # Adjust image dimensions to match the aspect ratio    
    
    #if img_width > img_height:
    #    img_width = img_width //2    
    #    img_height = int(img_width / aspect_ratio)
    #else:
        #img_width = img_width //2
        #img_height = int(img_height / aspect_ratio)
        
    plot_height, plot_width = img.shape[:2]
    x_range = Range1d(start=lbl_data['west_lon'], end=lbl_data['east_lon'])
    y_range = Range1d(start=lbl_data['min_lat'], end=lbl_data['max_lat'])

    p = figure(x_range=x_range,
               y_range=y_range,
               match_aspect=True,
               x_axis_label="Longitude (degrees)",
               y_axis_label="Latitude (degrees)",
               tools=['wheel_zoom,pan,box_zoom,reset'],
               width=plot_width, height=plot_height)

    p_vis = figure(x_range=p.x_range,  # Share x_range with p
                   y_range=p.y_range,  # Share y_range with p
                   match_aspect=True,
                   x_axis_label="Longitude (degrees)",
                   y_axis_label="Latitude (degrees)",
                   tools=['wheel_zoom,pan,box_zoom,reset'],
                   width=plot_width, height=plot_height)

    # Add the RGBA image to the plot
    p.image_rgba(image='image_rgba', source=source_image_rgba, x='x', y='y', dw='dw', dh='dh')
    p_vis.image_rgba(image='image_rgba', source=source_image_rgba2, x='x', y='y', dw='dw', dh='dh')

    
    # Update the ranges directly in p and p_vis
    p.x_range.start = lbl_data['west_lon']
    p.x_range.end = lbl_data['east_lon']
    p.y_range.start = lbl_data['min_lat']
    p.y_range.end = lbl_data['max_lat']

    # Define the new ticks based on the lon/lat in lbl_data
    longitude_ticks = list(np.arange(round(lbl_data['west_lon'], 1), round(lbl_data['east_lon'], 1), 0.05))
    latitude_ticks = list(np.arange(round(lbl_data['min_lat'], 1), round(lbl_data['max_lat'], 1), 0.05))

    # Update the tickers
    p.xaxis.ticker = FixedTicker(ticks=longitude_ticks)
    p.yaxis.ticker = FixedTicker(ticks=latitude_ticks)

    # Ensure grid lines are rendered on top using Span
    # Remove any existing Span objects (grid lines) if necessary
    p.renderers = [r for r in p.renderers if not isinstance(r, Span)]
    p_vis.renderers = [r for r in p_vis.renderers if not isinstance(r, Span)]

    # Add vertical (longitude) grid lines as Span objects
    for x in longitude_ticks:
        vline = Span(location=x, dimension='height', line_color='gray', line_dash='dashed', line_width=1, line_alpha=0.9)
        p.add_layout(vline)
        p_vis.add_layout(vline)

    # Add horizontal (latitude) grid lines as Span objects
    for y in latitude_ticks:
        hline = Span(location=y, dimension='width', line_color='gray', line_dash='dashed', line_width=1, line_alpha=0.9)
        p.add_layout(hline)
        p_vis.add_layout(hline)
    return p, p_vis, plot_height, plot_width



# Initialize variables
img, img_sr, wavelength, sr_names, lbl_data, hdr_folder_path = load_files(if_files[0])
nan_img = np.copy(img[:,:,:])
nan_img_sr_ma = np.ma.masked_values(nan_img, 65535)


# Define initial channels for RGB
#initial_channels = ["BD1900_2", "BD1500_2", "BD1435"]
initial_channels = band_dict['ICE']
vis_channels = band_dict['TRUE']
# Initial processing of the selected file
reference_RGB, nan_img_sr_ma = process_image(img, img_sr, initial_channels)
reference_RGB2, nan_img_sr_ma2 = process_image(img, img_sr, vis_channels)

# Convert RGB to RGBA and flip vertically to correct the flipped image
def convert_to_rgba(rgb_array):
    rgb_array = (rgb_array * 255).astype(np.uint8)
    rgb_array = np.flipud(rgb_array)  # Flip vertically to correct the upside-down image
    rgba_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2RGBA)
    rgba_array = np.ascontiguousarray(rgba_array)
    return np.frombuffer(rgba_array, dtype=np.uint32).reshape(rgba_array.shape[:2])

# Convert reference RGB to RGBA for display

image_rgba = convert_to_rgba(reference_RGB)
image_rgba2 = convert_to_rgba(reference_RGB2)

# Create a ColumnDataSource with the RGBA image and metadata for coordinates
source_image_rgba = ColumnDataSource(data={
    'image_rgba': [image_rgba],
    'x': [lbl_data['west_lon']],
    'y': [lbl_data['min_lat']],
    'dw': [lbl_data['east_lon'] - lbl_data['west_lon']],
    'dh': [lbl_data['max_lat'] - lbl_data['min_lat']]
})


source_image_rgba2 = ColumnDataSource(data={
    'image_rgba': [image_rgba2],
    'x': [lbl_data['west_lon']],
    'y': [lbl_data['min_lat']],
    'dw': [lbl_data['east_lon'] - lbl_data['west_lon']],
    'dh': [lbl_data['max_lat'] - lbl_data['min_lat']]
})


# Get the height and width of the image
#img_height, img_width = img.shape[:2]


#plot_width = 1600
#plot_height = 720

p, p_vis, plot_height, plot_width = plotter(lbl_data,img)#img_height, img_width)
print(lbl_data)
#tickergridder([p, p_vis], lbl_data)

# Band profile plot
p_band_profile = figure(width=plot_width*2, height=300, title="Spectrum", 
                        x_axis_label="Wavelength (μm)", y_axis_label="Reflectance", )
p_band_profile.line(x='Wavelength', y='Reflectance', source=band_profile_source, line_color='blue', line_width=2)#, legend_label="Band Profile")

# Function to convert latitude/longitude to pixel coordinates
def latlon_to_pixel(lon, lat, lbl_data):
    lon_range = lbl_data['east_lon'] - lbl_data['west_lon']
    lat_range = lbl_data['max_lat'] - lbl_data['min_lat']
    
    # Convert the lon/lat into pixel coordinates based on image dimensions
    x_pixel = int((lon - lbl_data['west_lon']) / lon_range * (plot_width - 1))
    y_pixel = int((lat - lbl_data['min_lat']) / lat_range * (plot_height - 1))
    
    return x_pixel, y_pixel

# Plot the band profile as a line

# Callback to update the band profile on mouse move
def update_band_profile(event):
    lon = event.x
    lat = event.y
    
    # Convert lat/lon to pixel coordinates
    x_pixel, y_pixel = latlon_to_pixel(lon, lat, lbl_data)
    
    if 0 <= x_pixel < plot_width and 0 <= y_pixel < plot_height:
        band_profile = nan_img_sr_ma[y_pixel, x_pixel, :]
        #band_profile_source.data = dict(Wavelength=np.arange(len(band_profile)), Reflectance=band_profile)
        band_profile_source.data = dict(Wavelength=wavelength, Reflectance=band_profile)

p.on_event('mousemove', update_band_profile)
p_vis.on_event('mousemove', update_band_profile)


# Callback to update the image when RGB channels are changed
def update_rgb_image(attr, old, new):    
    
    # Get selected RGB channels
    channels_names = [dropdown_r.value, dropdown_g.value, dropdown_b.value]
    
    # Process image again with selected channels
    reference_RGB, nan_img_sr_ma = process_image(img, img_sr, channels_names)
    
    # Update the RGBA image
    #image_rgba = convert_to_rgba(reference_RGB)
    # Convert the updated image to RGBA format
    updated_image_rgba = convert_to_rgba(reference_RGB)
    
    # Update the ColumnDataSource with the new image
    source_image_rgba.data = {
        'image_rgba': [updated_image_rgba],
        'x': [lbl_data['west_lon']],
        'y': [lbl_data['min_lat']],
        'dw': [lbl_data['east_lon'] - lbl_data['west_lon']],
        'dh': [lbl_data['max_lat'] - lbl_data['min_lat']]
    }    
    # Update the image in the plot
   # p.image_rgba(image=[image_rgba], x=lbl_data['west_lon'], y=lbl_data['min_lat'],
#dw=lbl_data['east_lon'] - lbl_data['west_lon'], dh=lbl_data['max_lat'] - lbl_data['min_lat'])

# Callback to update the image when RGB channels are changed
def update_rgb_image2(attr, old, new):    
    # Get selected RGB channels
    channels_names2 = [dropdown_r2.value, dropdown_g2.value, dropdown_b2.value]
    
    # Process image again with selected channels
    reference_RGB2, nan_img_sr_ma2 = process_image(img, img_sr, channels_names2)
    
    # Update the RGBA image
    #image_rgba = convert_to_rgba(reference_RGB)
    # Convert the updated image to RGBA format
    updated_image_rgba2 = convert_to_rgba(reference_RGB2)
    
    # Update the ColumnDataSource with the new image
    source_image_rgba2.data = {
        'image_rgba': [updated_image_rgba2],
        'x': [lbl_data['west_lon']],
        'y': [lbl_data['min_lat']],
        'dw': [lbl_data['east_lon'] - lbl_data['west_lon']],
        'dh': [lbl_data['max_lat'] - lbl_data['min_lat']]
    }
    #plot_width, plot_height = nan_img_sr_ma2.shape[:2]
    #p_vis.width = plot_width
    #p_vis.height = plot_height
# Initialize the RGB dropdowns

dropdown_r = Select(title="Red Channel", value="BD1900_2", options=sr_names)
dropdown_g = Select(title="Green Channel", value="BD1500_2", options=sr_names)
dropdown_b = Select(title="Blue Channel", value="BD1435", options=sr_names)

# Initialize the dropdown to select from band_dict
band_select = Select(title="Summary Product", value="ICE", options=list(band_dict.keys()))


dropdown_r2 = Select(title="Red Channel", value="BD1900_2", options=sr_names)
dropdown_g2 = Select(title="Green Channel", value="BD1500_2", options=sr_names)
dropdown_b2 = Select(title="Blue Channel", value="BD1435", options=sr_names)

# Initialize the dropdown to select from band_dict
band_select2 = Select(title="Summary Product", value="TRUE", options=list(band_dict.keys()))



# Update RGB dropdowns based on the selected band group
def update_rgb_channels(attr, old, new):
    selected_key = band_select.value
    if selected_key in band_dict:
        channels = band_dict[selected_key]
        # Update the RGB dropdowns with the selected band's RGB values
        dropdown_r.value = channels[0]
        dropdown_g.value = channels[1]
        dropdown_b.value = channels[2]


def update_rgb_channels2(attr, old, new):
    selected_key2 = band_select2.value
    if selected_key2 in band_dict:
        channels2 = band_dict[selected_key2]
        # Update the RGB dropdowns with the selected band's RGB values
        dropdown_r2.value = channels2[0]
        dropdown_g2.value = channels2[1]
        dropdown_b2.value = channels2[2]

# Dropdown callback to load a new I/F file and its SR and LBL counterpart
from bokeh.models import FixedTicker

from bokeh.models import Span, FixedTicker

def update_files(attr, old, new):
    global img, img_sr, reference_RGB, nan_img_sr_ma, lbl_data, wavelength, plot_height,plot_width,p,p_vis
    img, img_sr, wavelength, sr_names, lbl_data, hdr_folder_path= load_files(new)
    
    if img is None or img_sr is None or lbl_data is None:
        return
    
    # Update the RGB options with the new file's spectral parameters
    dropdown_r.options = [(sr_name, sr_name) for sr_name in sr_names]
    dropdown_g.options = [(sr_name, sr_name) for sr_name in sr_names]
    dropdown_b.options = [(sr_name, sr_name) for sr_name in sr_names]
    
    # Reset dropdown values to the default selected channels
    dropdown_r.value = initial_channels[0]
    dropdown_g.value = initial_channels[1]
    dropdown_b.value = initial_channels[2]

    # Process the new file with the initial channels
    reference_RGB, nan_img_sr_ma = process_image(img, img_sr, initial_channels)

    # Convert the updated image to RGBA format
    updated_image_rgba = convert_to_rgba(reference_RGB)
    
    # Update the ColumnDataSource with the new image
    source_image_rgba.data = {
        'image_rgba': [updated_image_rgba],
        'x': [lbl_data['west_lon']],
        'y': [lbl_data['min_lat']],
        'dw': [lbl_data['east_lon'] - lbl_data['west_lon']],
        'dh': [lbl_data['max_lat'] - lbl_data['min_lat']]
    }

    # Update the ranges directly in p and p_vis
    p.x_range.start = lbl_data['west_lon']
    p.x_range.end = lbl_data['east_lon']
    p.y_range.start = lbl_data['min_lat']
    p.y_range.end = lbl_data['max_lat']
    plot_height, plot_width, _ = img.shape
    print(img.shape)
    print('SIIIIZE: ', plot_height, plot_width)
    #p, p_vis, plot_height, plot_width = plotter(lbl_data,img)#img_height, img_width)
    #p.width = plot_width
    #p.height = plot_height
    #p_vis.width = plot_width
    #p_vis.height = plot_height
    # Define the new ticks based on the lon/lat in lbl_data
    longitude_ticks = list(np.arange(round(lbl_data['west_lon'], 1), round(lbl_data['east_lon'], 1), 0.05))
    latitude_ticks = list(np.arange(round(lbl_data['min_lat'], 1), round(lbl_data['max_lat'], 1), 0.05))

    # Update the tickers
    p.xaxis.ticker = FixedTicker(ticks=longitude_ticks)
    p.yaxis.ticker = FixedTicker(ticks=latitude_ticks)

    # Ensure grid lines are rendered on top using Span
    # Remove any existing Span objects (grid lines) if necessary
    p.renderers = [r for r in p.renderers if not isinstance(r, Span)]
    p_vis.renderers = [r for r in p_vis.renderers if not isinstance(r, Span)]
    
    # Add vertical (longitude) grid lines as Span objects
    for x in longitude_ticks:
        vline = Span(location=x, dimension='height', line_color='gray', line_dash='dashed', line_width=1, line_alpha=0.9)
        p.add_layout(vline)
        p_vis.add_layout(vline)

    # Add horizontal (latitude) grid lines as Span objects
    for y in latitude_ticks:
        hline = Span(location=y, dimension='width', line_color='gray', line_dash='dashed', line_width=1, line_alpha=0.9)
        p.add_layout(hline)
        p_vis.add_layout(hline)




def update_files2(attr, old, new):
    #global img, img_sr, wavelength, sr_names, lbl_data, hdr_folder_path
    img, img_sr, wavelength, sr_names, lbl_data, hdr_folder_path= load_files(new)
    
    if img is None or img_sr is None or lbl_data is None:
        return
    
    # Update the RGB options with the new file's spectral parameters
    dropdown_r2.options = [(sr_name, sr_name) for sr_name in sr_names]
    dropdown_g2.options = [(sr_name, sr_name) for sr_name in sr_names]
    dropdown_b2.options = [(sr_name, sr_name) for sr_name in sr_names]
    
    # Reset dropdown values to the default selected channels
    dropdown_r2.value = vis_channels[0]
    dropdown_g2.value = vis_channels[1]
    dropdown_b2.value = vis_channels[2]

    # Process the new file with the initial channels
    reference_RGB2, nan_img_sr_ma2 = process_image(img, img_sr, vis_channels)
    #image_rgba = convert_to_rgba(reference_RGB)
    
    # Update the image in the plot
    #p.image_rgba(image=[image_rgba], x=lbl_data['west_lon'], y=lbl_data['min_lat'],
    #             dw=lbl_data['east_lon'] - lbl_data['west_lon'], dh=lbl_data['max_lat'] - lbl_data['min_lat'])

    # Convert the updated image to RGBA format
    updated_image_rgba2 = convert_to_rgba(reference_RGB2)
    
    # Update the ColumnDataSource with the new image
    source_image_rgba2.data = {
        'image_rgba': [updated_image_rgba2],
        'x': [lbl_data['west_lon']],
        'y': [lbl_data['min_lat']],
        'dw': [lbl_data['east_lon'] - lbl_data['west_lon']],
        'dh': [lbl_data['max_lat'] - lbl_data['min_lat']]
    }
    
    # Update the ranges directly in p and p_vis
    p_vis.x_range.start = lbl_data['west_lon']
    p_vis.x_range.end = lbl_data['east_lon']
    p_vis.y_range.start = lbl_data['min_lat']
    p_vis.y_range.end = lbl_data['max_lat']
    #plot_width, plot_height = nan_img_sr_ma2.shape[:2]
    #p_vis.width = plot_width
    #p_vis.height = plot_height
    # Define the new ticks based on the lon/lat in lbl_data
    longitude_ticks = list(np.arange(round(lbl_data['west_lon'], 1), round(lbl_data['east_lon'], 1), 0.05))
    latitude_ticks = list(np.arange(round(lbl_data['min_lat'], 1), round(lbl_data['max_lat'], 1), 0.05))

    # Update the tickers
    p_vis.xaxis.ticker = FixedTicker(ticks=longitude_ticks)
    p_vis.yaxis.ticker = FixedTicker(ticks=latitude_ticks)


def update_pt_table(attr, old, new):
    
    
    xs = [i for i in p1.data_source.data['x']]
    ys = [i for i in p1.data_source.data['y']]
    #print('xs', xs)
    #print('ys', ys)
    print(p1.data_source.data['y'])
    #print('SOURCE DATA',p1.data_source.data)
    ids = list(np.arange(0,len(xs)))
    #types = ['' for el in ids
    band_profiles = []
    #lista = [latlon_to_pixel(lon, lat, lbl_data) for lon,lat in list(zip(p1.data_source.data['x'],p1.data_source.data['y']))]
    #for lon, lat in list(zip(p1.data_source.data['x'],p1.data_source.data['y'])):        
    #    x_pixel, y_pixel = latlon_to_pixel(lon, lat, lbl_data)
    #    if 0 <= x_pixel < width and 0 <= y_pixel < height:
            #band_profiles.append(nan_img_sr_ma[y_pixel, x_pixel, :])
    #print(lista)
    # Update the table data source
    source_table.data = {'ID':ids,'Lon': xs, 'Lat': ys}#, 'Spectrum': band_profiles}     
    #print(source_table.data)

def update_pt_table2(attr, old, new):
    
    
    xs2 = [i for i in p2.data_source.data['x']]
    ys2 = [i for i in p2.data_source.data['y']]
    #print('xs', xs)
    #print('ys', ys)
    print(p2.data_source.data['y'])
    #print('SOURCE DATA',p1.data_source.data)
    ids2 = list(np.arange(0,len(xs2)))
    #types = ['' for el in ids
    band_profiles2 = []
    #lista = [latlon_to_pixel(lon, lat, lbl_data) for lon,lat in list(zip(p1.data_source.data['x'],p1.data_source.data['y']))]
    #for lon, lat in list(zip(p1.data_source.data['x'],p1.data_source.data['y'])):        
    #    x_pixel, y_pixel = latlon_to_pixel(lon, lat, lbl_data)
    #    if 0 <= x_pixel < width and 0 <= y_pixel < height:
            #band_profiles.append(nan_img_sr_ma[y_pixel, x_pixel, :])
    #print(lista)
    # Update the table data source
    source_table2.data = {'ID':ids2,'Lon': xs2, 'Lat': ys2}#, 'Spectrum': band_profiles}     
    #print(source_table.data)

table_width=plot_width//4
table_height=plot_height//6


##################################### ROI
table_data = {'ID':[],'Lat': [], 'Lon': [], 'Spectrum': []}
source_table = ColumnDataSource(data=table_data)
source_table.data = table_data

xs = [table_data['Lon']]
ys = [table_data['Lat']]

source_point_draw = ColumnDataSource(data=dict(x=xs, y=ys))
source_point_draw.data = dict(x=xs, y=ys)

p1= p.scatter(x='x', y='y', source = source_point_draw, color='lightyellow',  size=15, marker='circle_cross')          
draw_tool_p1 = PointDrawTool(renderers=[p1])
p.add_tools(draw_tool_p1)

#p2= p.scatter(x='x', y='y', source = source_point_draw2, color='yellow',  size=20, marker='y')          
#draw_tool_p2 = PointDrawTool(renderers=[p2])
#p.add_tools(draw_tool_p2)



callback_p1 = CustomJS(args=dict(src=source_point_draw), code="""
    const data = cb_obj.data;
    src.data = data;
    src.change.emit();
""")

callback_table = CustomJS(args=dict(src=source_table), code="""
    const data = cb_obj.data;
    src.data = data;
    src.change.emit();
""")

columns = [
    TableColumn(field="ID", title="ID"),
    TableColumn(field="Lon", title="Longitude"),
    TableColumn(field="Lat", title="Latitude"),
    TableColumn(field="Spectrum", title="Spectrum"),
]

data_table = DataTable(source=source_table, columns=columns,editable=True, width=table_width, height=table_height, autosize_mode='fit_columns')

##################################### NEUTRAL
table_data2 = {'ID':[],'Lat': [], 'Lon': [], 'Spectrum': []}
source_table2 = ColumnDataSource(data=table_data2)
source_table2.data = table_data2

xs2 = [table_data2['Lon']]
ys2 = [table_data2['Lat']]

source_point_draw2= ColumnDataSource(data=dict(x=xs2, y=ys2))
source_point_draw2.data = dict(x=xs2, y=ys2)

p2= p.scatter(x='x', y='y', source = source_point_draw2, color='lightgreen',  size=15, marker='diamond')          
draw_tool_p2 = PointDrawTool(renderers=[p2])
p.add_tools(draw_tool_p2)

#p2= p.scatter(x='x', y='y', source = source_point_draw2, color='yellow',  size=20, marker='y')          
#draw_tool_p2 = PointDrawTool(renderers=[p2])
#p.add_tools(draw_tool_p2)



callback_p2 = CustomJS(args=dict(src=source_point_draw2), code="""
    const data = cb_obj.data;
    src.data = data;
    src.change.emit();
""")

callback_table2 = CustomJS(args=dict(src=source_table2), code="""
    const data = cb_obj.data;
    src.data = data;
    src.change.emit();
""")

columns = [
    TableColumn(field="ID", title="ID"),
    TableColumn(field="Lon", title="Longitude"),
    TableColumn(field="Lat", title="Latitude"),
    TableColumn(field="Spectrum", title="Spectrum"),
]

data_table2 = DataTable(source=source_table2, columns=columns,editable=True, width=table_width, height=table_height, autosize_mode='fit_columns')



# Link the I/F file dropdown to the callback
dropdown_if.on_change('value', update_files)
# Link RGB dropdowns to the callback
dropdown_r.on_change('value', update_rgb_image)
dropdown_g.on_change('value', update_rgb_image)
dropdown_b.on_change('value', update_rgb_image)
# Link the band group dropdown to the callback
band_select.on_change('value', update_rgb_channels)

dropdown_if.on_change('value', update_files2)
# Link RGB dropdowns to the callback
dropdown_r2.on_change('value', update_rgb_image2)
dropdown_g2.on_change('value', update_rgb_image2)
dropdown_b2.on_change('value', update_rgb_image2)
# Link the band group dropdown to the callback
band_select2.on_change('value', update_rgb_channels2)

source_point_draw.js_on_change("data", callback_p1)   
source_table.js_on_change("data", callback_table)
p1.data_source.on_change('data', update_pt_table)

source_point_draw2.js_on_change("data", callback_p2)   
source_table2.js_on_change("data", callback_table2)
p2.data_source.on_change('data', update_pt_table2)


# Initialize the band profile plot
band_profile_plot = figure(width=plot_width, height=300, title="ROI points Spectra",
                           x_axis_label="Wavelength (μm)", y_axis_label="Reflectance")

# Data source for band profiles
source_band_profiles = ColumnDataSource(data=dict(band=[], profile=[]))

# Data source for the mean band profile
source_mean_band_profile = ColumnDataSource(data=dict(band=[], mean_profile=[]))

# Plot individual band profiles
band_profile_plot.multi_line(xs='band', ys='profile', source=source_band_profiles, line_color='blue', alpha=0.6)

# Plot the mean band profile
band_profile_plot.line(x='band', y='mean_profile', source=source_mean_band_profile, line_color='red', line_width=3)#, legend_label='Mean Profile')

# Update band profiles and mean plot based on table data
# Update band profiles and mean plot based on table data
def update_band_profiles():
    xs = source_table.data['Lon']
    ys = source_table.data['Lat']
    
    # Flatten xs and ys to ensure they are simple lists
    xs = [item for sublist in xs for item in (sublist if isinstance(sublist, list) else [sublist])]
    ys = [item for sublist in ys for item in (sublist if isinstance(sublist, list) else [sublist])]
    
    band_profiles = []
    mean_profile = None
    
    # Extract band profiles from nan_img_sr_ma based on lat/lon
    for lon, lat in zip(xs, ys):
        x_pixel, y_pixel = latlon_to_pixel(lon, lat, lbl_data)
        if 0 <= x_pixel < plot_width and 0 <= y_pixel < plot_height:
            band_profile = nan_img_sr_ma[y_pixel, x_pixel, :]
            if band_profile.mean() > 0:
                band_profiles.append(list(band_profile))
    
    # If we have profiles, calculate mean profile
    if band_profiles:
        mean_profile = np.mean(band_profiles, axis=0)
        mean_profile = list(mean_profile)  # Convert to list for Bokeh
        
    # Update the data source for individual band profiles
    if band_profiles:
        #source_band_profiles.data = {'band': [list(range(len(band_profiles[0])))] * len(band_profiles), 'profile': band_profiles}
        source_band_profiles.data = {'band': [list(wavelength)] * len(band_profiles), 'profile': band_profiles}
    else:
        source_band_profiles.data = {'band': [], 'profile': []}
    
    # Update the data source for the mean band profile
    if mean_profile is not None:
        #source_mean_band_profile.data = {'band': list(range(len(mean_profile))), 'mean_profile': mean_profile}
        #source_mean_band_profile.data = {'band': wavelength, 'profile': mean_profile}
        # Check if wavelength and mean_profile have the same length
        if len(wavelength) == len(mean_profile):
            # Make sure to use consistent key names
            source_mean_band_profile.data = {'band': wavelength, 'mean_profile': mean_profile}
        else:
            print("Error: Wavelength and profile data lengths do not match.")

    else:
        source_mean_band_profile.data = {'band': [], 'mean_profile': []}

source_table.on_change('data', lambda attr, old, new: update_band_profiles())

# Initialize the band profile plot NEUTRAL
band_profile_plot2 = figure(width=plot_width, height=300, title="Neutral Spectra",
                           x_axis_label="Wavelength (μm)", y_axis_label="Reflectance",x_range=band_profile_plot.x_range,y_range=band_profile_plot.y_range)

# Data source for band profiles
source_band_profiles2 = ColumnDataSource(data=dict(band=[], profile=[]))

# Data source for the mean band profile
source_mean_band_profile2 = ColumnDataSource(data=dict(band=[], mean_profile=[]))

# Plot individual band profiles
band_profile_plot2.multi_line(xs='band', ys='profile', source=source_band_profiles2, line_color='blue', alpha=0.6)

# Plot the mean band profile
band_profile_plot2.line(x='band', y='mean_profile', source=source_mean_band_profile2, line_color='red', line_width=3)#, legend_label='Mean Profile')

# Update band profiles and mean plot based on table data
def update_band_profiles2():
    xs = source_table2.data['Lon']
    ys = source_table2.data['Lat']
    
    # Flatten xs and ys to ensure they are simple lists
    xs = [item for sublist in xs for item in (sublist if isinstance(sublist, list) else [sublist])]
    ys = [item for sublist in ys for item in (sublist if isinstance(sublist, list) else [sublist])]
    
    band_profiles2 = []
    mean_profile2 = None
    
    # Extract band profiles from nan_img_sr_ma based on lat/lon
    for lon, lat in zip(xs, ys):
        x_pixel, y_pixel = latlon_to_pixel(lon, lat, lbl_data)
        if 0 <= x_pixel < plot_width and 0 <= y_pixel < plot_height:
            band_profile2 = nan_img_sr_ma[y_pixel, x_pixel, :]
            if band_profile2.mean() > 0:
                band_profiles2.append(list(band_profile2))
    
    # If we have profiles, calculate mean profile
    if band_profiles2:
        mean_profile2 = np.mean(band_profiles2, axis=0)
        mean_profile2 = list(mean_profile2)  # Convert to list for Bokeh
        
    # Update the data source for individual band profiles
    if band_profiles2:
        #source_band_profiles2.data = {'band': [list(range(len(band_profiles2[0])))] * len(band_profiles2), 'profile': band_profiles2}
        source_band_profiles2.data = {'band': [list(wavelength)] * len(band_profiles2), 'profile': band_profiles2}
    else:
        source_band_profiles2.data = {'band': [], 'profile': []}
    
    # Update the data source for the mean band profile
    if mean_profile2 is not None:
        #source_mean_band_profile.data = {'band': list(range(len(mean_profile))), 'mean_profile': mean_profile}
        #source_mean_band_profile.data = {'band': wavelength, 'profile': mean_profile}
        # Check if wavelength and mean_profile have the same length
        if len(wavelength) == len(mean_profile2):
            # Make sure to use consistent key names
            source_mean_band_profile2.data = {'band': wavelength, 'mean_profile': mean_profile2}
        else:
            print("Error: Wavelength and profile data lengths do not match.")
    else:
        source_mean_band_profile2.data = {'band': [], 'mean_profile': []}


# Callback to update the plot when the table data changes
source_table2.on_change('data', lambda attr, old, new: update_band_profiles2())




# Initialize the ratio plot
ratio_plot = figure(width=plot_width*2, height=300, title="Ratio of Mean Band Profiles",
                    x_axis_label="Wavelength (μm)", y_axis_label="Ratio")

# Data source for the ratio plot
source_ratio_band_profile = ColumnDataSource(data=dict(band=[], ratio=[]))

# Plot the ratio
ratio_plot.line(x='band', y='ratio', source=source_ratio_band_profile, line_color='green', line_width=2)


# Initialize the second plot for normalized data
normalized_plot = figure(width=plot_width, height=300, title="Normalized Ratio of Mean Band Profiles",
                         x_axis_label="Wavelength (μm)", y_axis_label="Normalized Ratio")

# Data source for the normalized ratio plot
source_normalized_ratio_band_profile = ColumnDataSource(data=dict(band=[], normalized_ratio=[]))

# Plot the normalized ratio
normalized_plot.line(x='band', y='normalized_ratio', source=source_normalized_ratio_band_profile, line_color='blue', line_width=2)

# Function to normalize a list of values between 0 and 1 using MinMaxScaler
def normalize(values):
    if len(values) == 0:
        return []
    
    # Ensure values are a 2D numpy array
    values_array = np.array(values).reshape(-1, 1)
    
    # Initialize MinMaxScaler with the range (0, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Fit and transform the values
    normalized_values = scaler.fit_transform(values_array).flatten()
    
    return normalized_values

# Updated function to update both ratio and normalized ratio plots
def update_ratio():
    # Get the current mean profiles from both sources
    mean_profile1 = source_mean_band_profile.data.get('mean_profile', [])
    mean_profile2 = source_mean_band_profile2.data.get('mean_profile', [])
    
    # Ensure both profiles are non-empty and of the same length
    if len(mean_profile1) > 0 and len(mean_profile2) > 0 and len(mean_profile1) == len(mean_profile2):
        # Calculate the ratio between the two profiles
        ratio = [v1 / v2 if v2 != 0 else 0 for v1, v2 in zip(mean_profile1, mean_profile2)]
        
        # Debug: Print the ratio before normalization
        print("Ratio before normalization:", ratio)
        
        # Normalize the ratio between 0 and 1
        normalized_ratio = normalize(ratio)
        
        # Debug: Print normalized ratio
        print("Normalized Ratio:", normalized_ratio)
        
        # Update the data source for the original ratio plot
        source_ratio_band_profile.data = {'band': wavelength, 'ratio': ratio}
        
        # Update the data source for the normalized ratio plot
        source_normalized_ratio_band_profile.data = {'band': wavelength, 'normalized_ratio': normalized_ratio}
    else:
        # Clear the plots if the profiles are invalid or unequal in length
        source_ratio_band_profile.data = {'band': [], 'ratio': []}
        source_normalized_ratio_band_profile.data = {'band': [], 'normalized_ratio': []}

# Add the callback for whenever either source_mean_band_profile or source_mean_band_profile2 changes
source_mean_band_profile.on_change('data', lambda attr, old, new: update_ratio())
source_mean_band_profile2.on_change('data', lambda attr, old, new: update_ratio())








def list_tab_files_and_extract_base_names(folder_path):
    tab_files = []
    base_names = []

    for file in os.listdir(folder_path):
        if file.endswith(".tab"):
            # Extract the base name by removing the prefix 'crism_typespec_' and the '.tab' extension
            base_name = file.replace('crism_typespec_', '').replace('.tab', '')
            base_names.append(base_name)
            tab_files.append(file)
    
    return tab_files, base_names



# Get the list of .tab files and their corresponding base names
tab_files, base_names = list_tab_files_and_extract_base_names(spectra_library_path)

# Create a MultiSelect dropdown menu for base names
multi_select = MultiSelect(title="Select Data Files", value=[], options=base_names)
multi_select2 = MultiSelect(title="Select Data Files", value=[], options=base_names)

# Create an empty ColumnDataSource to be updated later
source_splib = ColumnDataSource(data={'name': [], 'wavelength': [], 'intensity': [], 'color': []})
source_splib2 = ColumnDataSource(data={'name': [], 'wavelength': [], 'intensity': [], 'color': []})
source_splib3 = ColumnDataSource(data={'name': [], 'wavelength': [], 'intensity': [], 'color': []})

from bokeh.palettes import Category10  # For generating a list of distinct colors

df_columns = ["WAVELENGTH", "RATIOED I/F CORRECTED", "RATIOED I/F", "I/F NUMERATOR CORRECTED", 
              "I/F NUMERATOR", "I/F DENOMINATOR CORRECTED", "I/F DENOMINATOR"]

def load_tab_files(selected_base_names):
    combined_data = {'name': [], 'wavelength': [], 'intensity': [], 'intensity_normalized': [], 'color': []}
    combined_data2 = {'name': [], 'wavelength': [], 'intensity': [], 'intensity_normalized': [], 'color': []}
    combined_data3 = {'name': [], 'wavelength': [], 'intensity': [], 'intensity_normalized': [], 'color': []}
    
    combined_wv_list, combined_intensity_list, combined_intensity_list2, combined_intensity_list3, combined_names, combined_colors,  = [], [], [], [], [], []
    
    # Generate a list of distinct colors using a Bokeh palette (e.g., Category10)
    color_palette = Category10[10]  # Can support up to 10 distinct colors; expand if needed

    for i, base_name in enumerate(selected_base_names):
        # Find the corresponding tab file
        index = base_names.index(base_name)
        tab_file = tab_files[index]
        tab_file_path = os.path.join(spectra_library_path, tab_file)

        # Read the .tab file (assume it's a comma-delimited file)
        data = pd.read_csv(tab_file_path, sep=',', header=None)
        data.columns = df_columns
        
        # Extract wavelength and intensity
        combined_wv_list.append(data[df_columns[0]].values)
        combined_intensity_list.append(data[df_columns[3]].values)
        combined_intensity_list2.append(data[df_columns[5]].values)
        combined_intensity_list3.append(data[df_columns[1]].values)

        # Initialize the MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))

        combined_intensity_list.append(data[df_columns[3]].values)
        combined_intensity_list2.append(data[df_columns[5]].values)
        combined_intensity_list3.append(data[df_columns[1]].values)


        combined_names.append(base_name)
        
        # Assign a color to this entry (cycling through color_palette)
        color = color_palette[i % len(color_palette)]  # Cycle through colors if more base names than colors
        combined_colors.append(color)

    # Normalize combined_intensity_list
        combined_intensity_list_normalized = [scaler.fit_transform(np.array(values).reshape(-1, 1)).flatten() for values in combined_intensity_list]

        # Normalize combined_intensity_list2
        combined_intensity_list2_normalized = [scaler.fit_transform(np.array(values).reshape(-1, 1)).flatten() for values in combined_intensity_list2]

        # Normalize combined_intensity_list3
        combined_intensity_list3_normalized = [scaler.fit_transform(np.array(values).reshape(-1, 1)).flatten() for values in combined_intensity_list3]


    # Combine the data for both intensity sets
    combined_data['wavelength'] = combined_wv_list        
    combined_data['intensity'] = combined_intensity_list
    combined_data['intensity_normalized'] = combined_intensity_list_normalized
    combined_data['name'] = combined_names
    
    combined_data['color'] = combined_colors  # Assign colors to the data
    
    combined_data2['wavelength'] = combined_wv_list
    combined_data2['intensity'] = combined_intensity_list2
    combined_data2['intensity_normalized'] = combined_intensity_list2_normalized
    combined_data2['name'] = combined_names
    combined_data2['color'] = combined_colors  # Assign the same colors to data2

    combined_data3['wavelength'] = combined_wv_list
    combined_data3['intensity'] = combined_intensity_list3
    combined_data3['intensity_normalized'] = combined_intensity_list3_normalized
    combined_data3['name'] = combined_names
    combined_data3['color'] = combined_colors  # Assign the same colors to data3

    # Update the ColumnDataSource
    source_splib.data = combined_data
    source_splib2.data = combined_data2
    source_splib3.data = combined_data3

# Create the plot and add HoverTool
hover = HoverTool(
    tooltips=[
        ("Name", "@name"),             # Show the name of the spectrum
        ("Wavelength", "$x (μm)"),          # Show the wavelength (x-coordinate)
        ("Intensity", "$y"),           # Show the intensity (y-coordinate)
    ]
)

# Add hover tool
band_profile_plot.add_tools(hover)
band_profile_plot2.add_tools(hover)
ratio_plot.add_tools(hover)

# When plotting, use the 'color' from the data source
band_profile_plot.multi_line('wavelength', 'intensity', source=source_splib, line_width=1, color='color')
band_profile_plot2.multi_line('wavelength', 'intensity', source=source_splib2, line_width=1, color='color')
ratio_plot.multi_line('wavelength', 'intensity', source=source_splib3, color='color', line_width=1)
normalized_plot.multi_line('wavelength', 'intensity_normalized', source=source_splib3, color='color', line_width=1)


# Callback function for when the dropdown selection changes
def dropdown_callback(attr, old, new):
    selected_base_names = multi_select.value
    load_tab_files(selected_base_names)

# Link the dropdown selection to the callback
multi_select.on_change('value', dropdown_callback)


# Export svgs
from bokeh.io import export_svg, export_png
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from bokeh.io.export import export_svgs

# Set up headless Chrome options
# Set up headless Chrome options




export_button_svg = Button(label="Export Spectral Plots as SVG")

def export_all_to_svg():

    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-gpu")

    # Set the path to the Chromium binary
    options.binary_location = "/usr/bin/chromium"

    # Set the path to chromedriver explicitly
    chrome_driver_path = "/usr/bin/chromedriver"

    # Create a Service instance with the path to chromedriver
    service = Service(executable_path=chrome_driver_path)

    # Create a browser instance using the service and options
    driver = webdriver.Chrome(service=service, options=options)

    label = dropdown_if.value
    # List of plots and their corresponding names for the filenames
    plots = [(band_profile_plot, 'numerator'), (band_profile_plot2, 'denominator')]
    

    savename = os.path.join(hdr_folder_path,label)
    # Loop through the list of plots and export each one
    for plt, name in plots:
        
        export_svg(plt, filename=f"{savename}_{name}.svg",webdriver=driver)
    driver.quit()

# Connect the single export function to the button
export_button_svg.on_click(export_all_to_svg)

layout = column(
    # Top row with dropdowns and p_band_profile
    row(subfolder_dropdown, dropdown_if, export_button_svg,multi_select),
    
    
    
    # Two columns below
    row(
        # Left column
        column(
            row(band_select, dropdown_r, dropdown_g, dropdown_b, ),
            p
        ),
        # Right column
        column(
            row(band_select2, dropdown_r2, dropdown_g2, dropdown_b2),
            p_vis
        ),
        column(
            row(p_band_profile),
        row(band_profile_plot, band_profile_plot2),
        row(ratio_plot),
        
    ),
    ),
    
    # Single column spanning both above columns
    
)



# Add the layout to the document
curdoc().clear()
curdoc().title = "CRISM-Toolkit:v01"  
curdoc().add_root(layout)