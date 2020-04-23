import numpy as np
# import bokeh
from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from bokeh.layouts import column
from bokeh.models import CustomJS, ColumnDataSource, Slider
from bokeh.plotting import Figure, output_file, show
from bokeh.models.callbacks import CustomJS
from bokeh.layouts import column
from bokeh.models import Button, CustomJS, ColumnDataSource, Slider, PointDrawTool
from bokeh.plotting import figure, output_file, show

from bokeh.plotting import figure, output_file, show
from bokeh.embed import components
from bokeh.embed import json_item
from bokeh.models import Panel, Tabs
from bokeh.models import ColumnDataSource, Label, LabelSet, Range1d


def distance(x1, y1, x2, y2):
    # compute cartesian distance
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)


def floorPlanPlotting(ld2, width, sects):

    width_height = int(width) - 25
    sects_cat = np.vstack((sects, sects[0, :]))
    tapX = sects_cat[:, 0:1]
    tapY = sects_cat[:, 1:2]

    ###########################
    plot = figure(plot_width=width_height, plot_height=width_height)
    
    plot1 = figure(plot_width=width_height, plot_height=width_height)
    plot2 = figure(plot_width=width_height, plot_height=width_height)

    ############################
    plot.toolbar.logo = None
    tools = "pan,wheel_zoom,reset,resize, hover"
    data = {'x_values': ld2.x,
            'y_values': ld2.y}

    source = ColumnDataSource(data=data)
    source1 = ColumnDataSource(data=dict(x=tapX, y=tapY))
    lidarPlot = plot.circle(x='x_values', y='y_values',
                            hover_color="red", source=source)

    customPlot = plot.scatter('x', 'y', source=source1,
                              line_color='red', fill_alpha=0.6, size=10)
    plot.line('x', 'y', source=source1, line_width=1,
              color="red", line_dash="4 4")
    draw_tool = PointDrawTool(renderers=[customPlot])

    callback = CustomJS(args=dict(sensor=source, xy=source1), code="""

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
    # callback.args["slider"] = rotation   # sending slider value into the callback function
    #rotation.js_on_change('value', callback)
    plot.js_on_event('tap', callback)
    layout = column(rotation, plot)   # create two row, 1. rotation 2. plotting
    plot.add_tools(draw_tool)
    plot.toolbar.active_tap = draw_tool

    tab = Panel(child=plot, title="Edit")

    ##################################
    x=[]
    y= []
    source2 = ColumnDataSource(data=dict(x=x, y=y))
    manualEditing = CustomJS(args=dict(manualSource=source2), code="""
            console.log("hi from edit")

            var data = manualSource.data;
            var manualX = data['x'];
            var manualY = data['y'];

            // getting slider value
            var rotationAngle = cb_obj.value;
            var radians = (Math.PI / 180) * rotationAngle;
            console.log("radians", radians);
            cos = Math.cos(radians);
            sin = Math.sin(radians);
            cx=2;
            cy=2;
            if(cb_obj.x) {
                    manualX.push(cb_obj.x);
                    manualY.push(cb_obj.y);
                    console.log(cb_obj.x);
                }
            
            }
            manualSource.change.emit();
            """)
    
    plot1.circle(x='x_values', y='y_values',hover_color="red", source=source)
    plot1.js_on_event('tap', manualEditing)
    plot1.add_tools(draw_tool)
    plot1.toolbar.active_tap = draw_tool
    plot1.patch('x', 'y',source=source2, line_width=10, color="red" )
    plot1.scatter(x='x', y='y', hover_color="red", source=source2, size=10)
    tab1 = Panel(child=plot1, title="Edit")


################################
   
    x = tapX.flatten()
    y = tapY.flatten()

    dist = []
    x_mid = []
    y_mid = []

    for i in range(0, len(x)-1):
            if (i != len(x)-1):
                dist1 = round(distance(x[i], y[i], x[i+1], y[i+1]))
                x_mid1 = (x[i] + x[i+1])/2
                y_mid1 = (y[i] + y[i+1])/2
            else:
                dist1 = round(distance(x[i], y[i], x[0], y[0]))
                x_mid1 = (x[i] + x[0])/2
                y_mid1 = (y[i] + y[0])/2
            dist.append(dist1)
            x_mid.append(x_mid1)
            y_mid.append(y_mid1)

    source = ColumnDataSource(data=dict(x_mid=x_mid, y_mid=y_mid, dist=dist))

    labels = LabelSet(x='x_mid', y='y_mid', text='dist', level='glyph',
                      x_offset=5, y_offset=5, source=source, render_mode='canvas', text_font_size="8pt")

    plot2.add_layout(labels)
    plot2.line('x', 'y', source=source1, line_width=1, color="red")
    tab2 = Panel(child=plot2, title="Draw floor Plan")

    tabs = Tabs(tabs=[tab,tab1, tab2])

    ##################################

    script_plot, div1 = components(tabs)
    responst_format = {
        'container': div1,
        'script': script_plot
    }

    return responst_format


'''
def plotting(kla):
    responst_format = dict()

    test = kla

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

    script_plot, div1 = components(p)
    responst_format = {
        'container': div1,
        'script': script_plot
    }
    return responst_format
'''
