buff = 0.5
# Add longitude (x) grid lines
for x in np.arange(round(lbl_data['west_lon']-buff,1), round(lbl_data['east_lon']+buff,1), 0.05):  # Adjust num for desired number of grid lines
    vline = Span(location=x, dimension='height', line_color='gray', line_dash='dashed', line_width=1, line_alpha=0.9)
    p.add_layout(vline)

# Add latitude (y) grid lines
for y in np.arange(round(lbl_data['min_lat']-buff,1), round(lbl_data['max_lat']+buff,1), 0.05):  # Adjust num for desired number of grid lines
    hline = Span(location=y, dimension='width', line_color='gray', line_dash='dashed', line_width=1, line_alpha=0.9)
    p.add_layout(hline)

# Set custom major ticks for x and y axes using FixedTicker
p.xaxis[0].ticker = FixedTicker(ticks=np.arange(round(lbl_data['west_lon']-buff,1), round(lbl_data['east_lon']+buff,1), 0.05))  # Set specific x-axis ticks
p.yaxis[0].ticker = FixedTicker(ticks=np.arange(round(lbl_data['min_lat']-buff,1), round(lbl_data['max_lat']+buff,1), 0.05))      # Set specific y-axis ticks

###################################################################################### VISIBLE PLOT

p_vis = figure(x_range=p.x_range,
                y_range=p.y_range,
                match_aspect=True,
                x_axis_label="Longitude (degrees)",
                y_axis_label="Latitude (degrees)",
                tools=[' wheel_zoom,pan,box_zoom,reset',cht],width=plot_width//4, height=plot_height)

# Add the RGBA image to the plot
#p.image_rgba(image=[image_rgba], x=lbl_data['west_lon'], y=lbl_data['min_lat'],
#             dw=lbl_data['east_lon'] - lbl_data['west_lon'], dh=lbl_data['max_lat'] - lbl_data['min_lat'])
p_vis.image_rgba(image='image_rgba', source=source_image_rgba2, x='x', y='y', dw='dw', dh='dh')

# Add grid lines at major ticks
#p.xgrid.grid_line_color = "gray"
#p.xgrid.grid_line_width = 1
#p.ygrid.grid_line_color = "gray"
#p.ygrid.grid_line_width = 1
from bokeh.models import Span, FixedTicker

buff = 0.5
# Add longitude (x) grid lines
for x in np.arange(round(lbl_data['west_lon']-buff,1), round(lbl_data['east_lon']+buff,1), 0.05):  # Adjust num for desired number of grid lines
    vline = Span(location=x, dimension='height', line_color='gray', line_dash='dashed', line_width=1, line_alpha=0.9)
    p_vis.add_layout(vline)

# Add latitude (y) grid lines
for y in np.arange(round(lbl_data['min_lat']-buff,1), round(lbl_data['max_lat']+buff,1), 0.05):  # Adjust num for desired number of grid lines
    hline = Span(location=y, dimension='width', line_color='gray', line_dash='dashed', line_width=1, line_alpha=0.9)
    p_vis.add_layout(hline)

# Set custom major ticks for x and y axes using FixedTicker
p_vis.xaxis[0].ticker = FixedTicker(ticks=np.arange(round(lbl_data['west_lon']-buff,1), round(lbl_data['east_lon']+buff,1), 0.05))  # Set specific x-axis ticks
p_vis.yaxis[0].ticker = FixedTicker(ticks=np.arange(round(lbl_data['min_lat']-buff,1), round(lbl_data['max_lat']+buff,1), 0.05))      # Set specific y-axis ticks


