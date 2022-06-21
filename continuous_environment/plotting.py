###
#  plotting: 
#  
#  (Multi)polygon plot functions



from shapely.geometry import Polygon, MultiPolygon, LineString
from matplotlib.axes import Axes


def plot_multipolygon(multiPolygon: MultiPolygon, color: str, axes: Axes):
    """Plots MultiPolygon in `color` to `axes`

    Args:
        multiPolygon (MultiPolygon): Shapes to plot
        color (str): Color to plot in, e.g. "blue"
        axes (_type_): Matplotlib axes to plot to
    """
    for polygon in multiPolygon.geoms:
        axes.fill(*polygon.exterior.xy, fc=color)


def plot_polygon(polygon: Polygon, color: str, axes: Axes):
    """Plots Polygon in `color` to `axes`

    Args:
        polygon (MultiPolygon): Shape to plot
        color (str): Color to plot in, e.g. "blue"
        axes (_type_): Matplotlib axes to plot to
    """
    axes.fill(*polygon.exterior.xy, fc=color)
