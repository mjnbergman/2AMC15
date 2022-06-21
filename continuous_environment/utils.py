###
#  utils:
#
#  parsing geometries functions


from shapely.geometry import LineString, Polygon

def parse_roomsize(roomsize: list) -> list:
    """ Parses room size to LineString representing outer bounding box. Used
        to parse room size.

    Args:
        roomsize (list[width, height]): Room size laoded form config file

    Returns:
        list[LineString]: List of length one containing LineString for outer
            boundary of room.
    """  
    return [LineString([
        (0, 0),                      # Bottom left
        (0, roomsize[1]),            # Top left
        (roomsize[0], roomsize[1]),  # Top right
        (roomsize[0], 0),            # Bottom right
        (0, 0)
    ])]      


def parse_polygons(multiPolygonCoords: list) -> list:
    """ Parses nested structure in config file to list of Polygons. Used to 
        parse goals, deaths, and walls.

    Args:
        multiPolygonCoords (list[list[list]]): 
            Structure: [ [[0, 0], [1, 0], ...] , ...]

    Returns:
        list[Polygon]: List of polygons
    """
    return [Polygon(polygonCoords) for polygonCoords in multiPolygonCoords]