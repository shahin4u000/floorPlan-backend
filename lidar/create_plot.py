import numpy as np
# import bokeh
from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from bokeh.layouts import column
from bokeh.models import CustomJS, ColumnDataSource, Slider
from bokeh.plotting import Figure, output_file, show
from bokeh.models.callbacks import CustomJS
from bokeh.layouts import column
from bokeh.models import Button, CustomJS, ColumnDataSource, Slider,PointDrawTool
from bokeh.plotting import figure, output_file, show

from bokeh.plotting import figure, output_file, show
from bokeh.embed import components
from bokeh.embed import json_item


def floorPlanPlotting(ld2):
    plot = figure(plot_width=400, plot_height=400)
    plot.toolbar.logo = None
    tapX = []
    tapY = []
    tools = "pan,wheel_zoom,reset,hover"
    data = {'x_values': ld2.x,
            'y_values': ld2.y}

    source = ColumnDataSource(data=data)
    source1 = ColumnDataSource(data=dict(x=tapX, y=tapY))
    lidarPlot= plot.circle(x='x_values', y='y_values',hover_color="red", source=source)

    customPlot= plot.scatter('x', 'y', source=source1,
                line_color='blue', fill_alpha=0.3, size=5)
    plot.patch('x', 'y',source=source1, line_width=5, color="red", alpha=0.5)
    draw_tool = PointDrawTool(renderers=[customPlot])

    callback = CustomJS(args=dict(sensor=source, xy=source1 ), code="""
                        
            // getting lidar value and tabX & tapY
            var data = xy.data;
            var sensorData = sensor.data;
            
            // getting slider value
            var rotationAngle = cb_obj.value;        
            var slider = rotationAngle * (Math.PI / 180);
                        
            console.log(slider)
            // 
            var x = data['x'];
            var y = data['y'];
            var a = sensorData['x_values'];
            var b = sensorData['y_values'];
            console.log(a);
            console.log(a.length);
            /*
            if(a.includes(cb_obj.x) && b.includes(cb_obj.y)) {
                x.push(cb_obj.x);
                y.push(cb_obj.y);
                
            } else {
                x.push(cb_obj.x);
                y.push(cb_obj.y);
                console.log(cb_obj.x);
            }
            */
            
            for (var i = 0; i < a.length; i++) {
                a[i] =  a[i]*Math.cos(slider) - b[i]*Math.sin(slider);
                b[i] =  a[i]*Math.sign(slider) + b[i]*Math.cos(slider); 
            }
            
            sensor.change.emit();
            xy.change.emit();
        """)

    rotation = Slider(start=-180, end=180, value=0, step=0.1, title="Rotate ")
    #callback.args["slider"] = rotation   # sending slider value into the callback function
    rotation.js_on_change('value', callback)
    plot.js_on_event('tap', callback)
    layout = column(rotation, plot)   # create two row, 1. rotation 2. plotting
    plot.add_tools(draw_tool)
    plot.toolbar.active_tap = draw_tool
    
    script_plot, div1=components(plot)
    responst_format={
        'container':div1,
        'script':script_plot
    }
    return responst_format
    
def plotting(kla):
    responst_format=dict()
    
    test=kla

    N = 4000
    x = np.random.random(size=N) * 100
    y = np.random.random(size=N) * 100
    radii = np.random.random(size=N) * 1.5
    colors = [
        "#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50+2*x, 30+2*y)]

    TOOLS = "hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"

    p = figure(tools=TOOLS)
    p.scatter(x, y, radius=radii,
                fill_color=colors, fill_alpha=0.6,
                line_color=None)

    script_plot, div1=components(p)
    responst_format={
        'container':div1,
        'script':script_plot
    }
    return responst_format
