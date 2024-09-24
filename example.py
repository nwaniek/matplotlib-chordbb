#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from chordbb import *

# Example usage
fig, ax = plt.subplots(figsize=(7,7))
ax.set_aspect('equal')

# Labels, quantities, and group assignments
labels = ['M', 'a', 'X', 'YL', 'longtext', 'f', 'hello']
quantities = [10, 5, 10, 17, 70, 3, 1]
# Group assignments for the segments
groups = [0, 0, 0, 1, 1, 2, 2]
# can be assigned for each element individually
radius = [0.3, 0.3, 0.3, 0.35, 0.35, 0.4, 0.4]
#radius = [0.3, 0.35, 0.4]

# Colors for each segment
color_list = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta']

# Gap widths (in degrees)
item_gap_width = np.deg2rad(3.0)
group_gap_width = 7 * item_gap_width

# Draw arc segments with gaps and groups
#

extent = (0, 2*np.pi)

# build the list of draw calls
render_steps = Renderers([
    ChordArcRenderer(linewidth=5),
    # ChordTickRenderer(),
    ChordArcAxisTicker(),
    ChordLabelRenderer(
        text_anchor='center',
        text_rotation_mode='left',
        text_displacement=0.01,
        fontsize=11,
        fontweight='normal'),
    ChordAnnulusSectorRenderer(),
    ])

segments = draw_chord_segments(ax, labels, quantities, groups,
                               item_gap_width, group_gap_width,
                               color_list,
                               render_steps,
                               extent=extent,
                               radius=radius,
                               radius_type='per_item')


arc_flow_renderer = ChordArcFlowRenderer(alpha=0.1, lw=0)
arc_flow_renderer(ax, segments[1], segments[6])
arc_flow_renderer(ax, segments[2], segments[5])
arc_flow_renderer(ax, segments[4], segments[2])
arc_flow_renderer(ax, segments[0], segments[3], alpha=0.4, lw=1)


# Remove axes for a clean look
ax.axis('off')

plt.show()

