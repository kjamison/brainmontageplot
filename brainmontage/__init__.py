#expose these functions
from .brainmontage import (create_montage_figure, retrieve_atlas_info, fill_surface_rois, fill_volume_rois, 
                           generate_surface_view_lookup, render_surface_lookup, save_atlas_lookup_file, load_atlas_lookup
)

from .utils import save_image

from ._version import __version__
