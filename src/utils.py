import json

from src.grid import Grid

def parse_config(filename: str) -> Grid:
    """ Convert grid json to Grid object for environment
    
    Args:
        filename (str): JSON file with keys `grid_size`, `obstacles`, and `goals`

    Returns:
        Grid: Specified environment
    """
    # Open JSON
    with open(filename) as f:
        data = json.load(f)
    
    # Initialize empty grid
    grid = Grid(*data["grid_size"])

    # Add goals
    for coords in data["goals"]:
        grid.put_goal(*coords)
        
    # Add obstacles
    for coords in data["obstacles"]:
        grid.put_obstacle(*coords)

    
    return grid