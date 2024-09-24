#!/usr/bin/env python

from typing import Literal, Callable, Tuple
import collections.abc
import numpy as np
from matplotlib.patches import Polygon


__version__ = '0.1.0'


def angle_to_coord(angle, center, radius):
    return (center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle))


def normalized_angular_distance(a, b):
    angular_distance = np.abs(a - b) % (2 * np.pi)
    if angular_distance > np.pi:
        angular_distance = 2 * np.pi - angular_distance
    normalized_distance = angular_distance / np.pi
    return normalized_distance


def get_bezier_vertices(a, b, radius_a, radius_b, vertex_count=100, center=(0.5, 0.5), base_scale=0.8):
    # start and end are based on the radius of the two circles for the chord
    # plot. Note that the radii can be different, which will lead to slightly
    # different control point anchors below
    start_point = np.array([center[0] + radius_a * np.cos(a), center[1] + radius_a * np.sin(a)])
    end_point = np.array([center[0] + radius_b * np.cos(b), center[1] + radius_b * np.sin(b)])

    # directions should be perpendicular to circle
    start_dir = -np.array([np.cos(a), np.sin(a)])
    end_dir   = -np.array([np.cos(b), np.sin(b)])

    # adjust scale factor based on distance
    angular_distance = normalized_angular_distance(a, b)
    angular_scale = np.sqrt(angular_distance)
    scale_factor_a = base_scale * angular_scale * radius_a
    scale_factor_b = base_scale * angular_scale * radius_b
    control_point1 = start_point + scale_factor_a * start_dir
    control_point2 = end_point + scale_factor_b* end_dir

    # finally generate bezier curve points
    t = np.linspace(0, 1, vertex_count)
    bezier_curve = ((1 - t[:, None])**3 * start_point +
                    3 * (1 - t[:, None])**2 * t[:, None] * control_point1 +
                    3 * (1 - t[:, None]) * t[:, None]**2 * control_point2 +
                    t[:, None]**3 * end_point)

    return bezier_curve


class AnnulusSector(Polygon):
    def __init__(self, center, inner_radius, outer_radius, theta1, theta2,
                 edgecolor: str | None = 'black',
                 facecolor='none',
                 linewidth=1,
                 vertex_count=100,
                 **kwargs):

        # create the vertices of the annulus sector
        vertices = self._get_vertices(center, inner_radius, outer_radius, theta1, theta2, vertex_count)

        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.theta1 = theta1
        self.theta2 = theta2

        # initialize the polygon
        super().__init__(vertices, closed=True, edgecolor=edgecolor, facecolor=facecolor, linewidth=linewidth, **kwargs)

    def _get_vertices(self, center, inner_radius, outer_radius, theta1, theta2, vertex_count):
        theta = np.linspace(theta1, theta2, vertex_count)

        # create outer and inner arc points
        outer_arc = np.array([center[0] + outer_radius * np.cos(t) for t in theta])
        outer_arc_y = np.array([center[1] + outer_radius * np.sin(t) for t in theta])
        inner_arc = np.array([center[0] + inner_radius * np.cos(t) for t in theta])
        inner_arc_y = np.array([center[1] + inner_radius * np.sin(t) for t in theta])

        # combine vertices to get annulus polygon
        outer_vertices = np.vstack((outer_arc, outer_arc_y)).T
        inner_vertices = np.vstack((inner_arc, inner_arc_y)).T
        vertices = np.concatenate((outer_vertices, inner_vertices[::-1]), axis=0)

        return vertices


class ChordBezierFlow(Polygon):
    def __init__(self,
                 arc1_start_angle: float, arc1_end_angle: float, arc1_radius: float,
                 arc2_start_angle: float, arc2_end_angle: float, arc2_radius: float,
                 vertex_count:int = 100,
                 center: Tuple[float, float] = (0.5, 0.5),
                 edgecolor='black',
                 facecolor='none',
                 linewidth=1,
                 **kwargs):
        vertices = self._get_vertices(center, arc1_start_angle, arc1_end_angle, arc1_radius,
                                       arc2_start_angle, arc2_end_angle, arc2_radius, vertex_count)
        super().__init__(vertices, closed=True, edgecolor=edgecolor, facecolor=facecolor, linewidth=linewidth, **kwargs)

    def _get_vertices(self, center, arc1_start_angle, arc1_end_angle, arc1_radius,
                       arc2_start_angle, arc2_end_angle, arc2_radius, vertex_count):
        # vertices for arc1
        theta1 = arc1_start_angle
        theta2 = arc1_end_angle
        theta = np.linspace(theta1, theta2, vertex_count)
        inner_arc1 = np.array([(center[0] + arc1_radius * np.cos(t), center[1] + arc1_radius * np.sin(t)) for t in theta])

        # vertices for arc2
        theta1 = arc2_start_angle
        theta2 = arc2_end_angle
        theta = np.linspace(theta1, theta2, vertex_count)
        inner_arc2 = np.array([(center[0] + arc2_radius * np.cos(t), center[1] + arc2_radius * np.sin(t)) for t in theta])

        # bezier curves that connect the two arcs
        bezier_curve0 = get_bezier_vertices(arc1_end_angle, arc2_start_angle, arc1_radius, arc2_radius)
        bezier_curve1 = get_bezier_vertices(arc2_end_angle, arc1_start_angle, arc2_radius, arc1_radius)

        # Ensure proper connection from end of inner_arc1 to start of inner_arc2
        vertices = np.concatenate([
            inner_arc1,
            bezier_curve0[1:-1],
            inner_arc2,
            bezier_curve1[1:-1],
        ], axis=0)

        return vertices


class Renderers(list):
    pass


def draw_chord_segments(ax,
               labels: list,
               quantities: list[float] | list[int],
               groups: list,
               item_gap_width: float,
               group_gap_width: float,
               color_list: list,
               render_steps: Renderers | list[Renderers],
               center: Tuple[float, float] = (0.5, 0.5),
               extent: Tuple[float, float] = (0.0, 2 * np.pi),
               radius: float | list[float] | np.ndarray = 0.4,
               radius_type: Literal['per_item', 'per_group'] = 'per_item'):
    """Function to draw arc segments around a circle with grouping and gaps"""

    n_labels = len(labels)

    # check number of groups, and if none is there, we default to 'group 0'
    # everywhere. this streamlines the code further below so that we don't need
    # to test for groups
    if groups is None:
        groups = [0] * n_labels
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    # the number of group and item gaps depends on the number of groups, and
    # thus also the actual angles consumed by the gaps
    n_group_gaps = n_groups if n_groups > 1 else 0
    n_item_gaps = n_labels if n_groups <= 1 else sum(1 for i in range(len(groups) - 1) if groups[i] == groups[i + 1])
    total_group_gap_angle = n_group_gaps * group_gap_width
    total_item_gap_angle = n_item_gaps * item_gap_width

    # need to remove the gaps from the overall extent that is available to have
    # sufficient space
    total_angle = extent[1] - extent[0]
    remaining_angle = total_angle - total_group_gap_angle - total_item_gap_angle
    total_quantity = sum(quantities)
    angle_per_unit = remaining_angle / total_quantity

    # figure out if radius is a sequence or not, and build an appropriate lambda
    # to avoid having this logic in the loop
    if isinstance(radius, (collections.abc.Sequence, np.ndarray)):
        if radius_type == 'per_item':
            if len(radius) != n_labels:
                raise ValueError("Non-scalar radius must have same length as labels in radius_mode 'per_item'")
            get_radius = lambda i, _: radius[i]
        elif radius_type == 'per_group':
            if len(radius) != n_groups:
                raise ValueError("Non-scalar radius must have same length as groups in radius_mode 'per_group'")
            get_radius = lambda _, g: radius[g]
        else:
            raise ValueError(f"unknown radius_type '{radius_type}'")
    else:
        get_radius = lambda _, __, radius=radius: radius


    current_angle = extent[0]

    segments = []
    for i, (quantity, group) in enumerate(zip(quantities, groups)):
        r = get_radius(i, group)

        # determine the start and end angle for this segment. This function does
        # not really plot anything itself, but prepares all the information for
        # the renderers
        segment_extent = quantity * angle_per_unit
        end_angle = current_angle + segment_extent
        center_angle = (current_angle + end_angle) / 2

        # this here can be used (and updated) by renderers
        segment = {
            'index':        i,
            'label':        labels[i],
            'quantity':     quantities[i],
            'radius':       r,
            'center':       center,
            'extent':       segment_extent,
            'start_angle':  current_angle,
            'end_angle':    end_angle,
            'center_angle': center_angle,
            'group':        group,
            'color':        color_list[i],
        }

        # we need to identify if the render_steps are a single renderer, or a
        # list of renderers, and act accordingly. Calls to renderers can update
        # the segment
        if isinstance(render_steps, Renderers):
            for renderer in render_steps:
                segment = renderer(ax, segment)
        elif isinstance(render_steps, list):
            for renderer in render_steps[i]:
                segment = renderer(ax, segment)
        elif callable(render_steps):
            segment = render_steps(ax, segment)
        else:
            raise ValueError(f"Cannot handle draw_calls of type '{type(render_steps)}'. Must be either Renderers, list, or Callable.")

        segments.append(segment)
        current_angle += segment_extent

        # add item gap if next item is in the same group
        if i < len(groups) - 1 and groups[i] == groups[i + 1]:
            current_angle += item_gap_width

        # add group gap if next item is in a different group
        elif i < len(groups) - 1 and groups[i] != groups[i + 1]:
            current_angle += group_gap_width

    return segments


class ChordArcRenderer:
    def __init__(self,
                 width = 0.0015,
                 displacement: Literal['auto'] | float | None = 'auto',
                 accumulate_width: bool = True,
                 **kwargs):
        self.width = width
        self.accumulate_width = accumulate_width
        self.kwargs = kwargs
        if displacement is None:
            self.get_displacement = lambda _ : 0.0
        elif isinstance(displacement, float):
            self.get_displacement = lambda _, d=displacement: d
        elif isinstance(displacement, str) and displacement == 'auto':
            self.get_displacement = lambda segment: segment.get('width', 0.0)

    def __call__(self, ax, segment, **kwargs):
        _kwargs = self.kwargs | kwargs

        center = segment['center']
        radius = segment['radius'] + self.get_displacement(segment)
        theta1 = segment['start_angle']
        theta2 = segment['end_angle']
        color  = _kwargs.pop('edgecolor', segment['color'])

        color='black'
        arc = AnnulusSector(center,
                            radius,
                            radius+self.width,
                            theta1, theta2,
                            edgecolor=None,
                            facecolor=color,
                            **self.kwargs)
        ax.add_patch(arc)
        segment['arc'] = arc

        accum = segment.get('width', 0.0)
        accum = accum + self.width if self.accumulate_width else accum
        segment['width'] = accum

        return segment


class ChordAnnulusSectorRenderer:
    def __init__(self,
                 width=0.05,
                 displacement: Literal['auto'] | float | None = 'auto',
                 accumulate_width: bool = True,
                 **kwargs):
        self.width = width
        self.accumulate_width = accumulate_width
        self.kwargs = kwargs

        if displacement is None:
            self.get_displacement = lambda _ : 0.0
        elif isinstance(displacement, float):
            self.get_displacement = lambda _, d=displacement: d
        elif isinstance(displacement, str) and displacement == 'auto':
            self.get_displacement = lambda segment: segment.get('width', 0.0)

    def __call__(self, ax, segment):
        r      = segment['radius'] + self.get_displacement(segment)
        center = segment['center']
        start  = segment['start_angle']
        end    = segment['end_angle']
        color  = segment['color']

        alpha = self.kwargs.pop('alpha', 0.5)
        edgecolor = self.kwargs.pop('edgecolor', None)
        facecolor = self.kwargs.pop('facecolor', color)

        annulus = AnnulusSector(center, r, r+self.width, start, end, edgecolor=edgecolor, facecolor=facecolor, alpha=alpha, **self.kwargs)
        ax.add_patch(annulus)

        accum = segment.get('width', 0.0)
        accum = accum + self.width if self.accumulate_width else accum
        segment['width'] = accum

        segment['annulus_sector'] = annulus
        segment['annulus_sector_kwargs'] = self.kwargs
        return segment


class ChordTicker:
    def __init__(self,
                 tick_length: float = 0.01,
                 tick_displacement: float | None | Literal['auto'] = 'auto',
                 accumulate_width: bool = True,
                 **kwargs):
        self.tick_length = tick_length
        self.accumulate_width = accumulate_width
        self.kwargs = kwargs

        if isinstance(tick_displacement, float):
            self.get_tick_displacement = lambda _, d=tick_displacement: d
        elif isinstance(tick_displacement, str) and tick_displacement == 'auto':
            self.get_tick_displacement = lambda segment: segment['width'] if 'width' in segment else 0.0
        elif tick_displacement is None:
            self.get_tick_displacement = lambda _ : 0.0
        else:
            raise ValueError(f"Invalid value '{tick_displacement} for tick_displacement")


    def __call__(self, ax, segment):
        r            = segment['radius']
        center       = segment['center']
        center_angle = segment['center_angle']
        disp         = self.get_tick_displacement(segment)

        x_start = center[0] + (r + disp) * np.cos(center_angle)
        y_start = center[1] + (r + disp) * np.sin(center_angle)
        x_end = center[0] + (r + self.tick_length + disp) * np.cos(center_angle)
        y_end = center[1] + (r + self.tick_length + disp) * np.sin(center_angle)
        tick_line = ax.plot([x_start, x_end], [y_start, y_end], color='black', linewidth=1, **self.kwargs)
        segment['tick'] = tick_line
        segment['tick_start'] = (x_start, y_start)
        segment['tick_end'] = (x_end, y_end)
        segment['tick_kwargs'] = self.kwargs

        accum = segment.get('width', 0.0)
        accum = accum + self.tick_length if self.accumulate_width else accum
        segment['width'] = accum

        return segment


class ChordArcAxisTicker:
    def __init__(self,
                 ticklabel_format="",
                 ticklabel_interval=10,
                 tick_lw=0.5,
                 tick_length=0.01,
                 special_tick_lw=1.5,
                 special_tick_length=0.015,
                 special_tick_interval=10):
        # TODO: use matplotlib's regular formatter functions
        self.ticklabel_format   = ticklabel_format
        self.ticklabel_interval = ticklabel_interval
        self.lw                 = tick_lw
        self.length             = tick_length
        self.slw                = special_tick_lw
        self.slength            = special_tick_length
        self.sinterval          = special_tick_interval


    def draw_tick(self, ax, r, center, disp, angle, length, **kwargs):
        x_start = center[0] + (r + disp) * np.cos(angle)
        y_start = center[1] + (r + disp) * np.sin(angle)
        x_end = center[0] + (r + length + disp) * np.cos(angle)
        y_end = center[1] + (r + length + disp) * np.sin(angle)
        return ax.plot([x_start, x_end], [y_start, y_end], color='black', **kwargs)

    def __call__(self, ax, segment):
        e = segment['extent']
        q = segment['quantity']
        c = segment['center']
        r = segment['radius'] + segment.get('width', 0.0)
        s = segment['start_angle']

        inter_tick_distance = 1.0
        tick_angles = e * np.arange(0, q, inter_tick_distance) / q + s

        for i, angle in enumerate(tick_angles):
            lw = self.lw
            length = self.length
            if self.sinterval > 0 and i % self.sinterval == 0:
                lw = self.slw
                length = self.slength
            self.draw_tick(ax, r, c, 0.0, angle, length, lw=lw)

            if i % self.ticklabel_interval == 0:
                va = 'bottom'
                ha = 'center'
                rmode = 'anchor'
                x_label = c[0] + (r + 0.02) * np.cos(angle)
                y_label = c[1] + (r + 0.02) * np.sin(angle)
                textobj = ax.text(x_label, y_label,
                                  f"{i}",
                                  va=va, ha=ha,
                                  rotation=np.rad2deg(angle-np.pi/2),
                                  rotation_mode=rmode)

        accum = segment.get('width', 0.0) + 0.055
        segment['width'] = accum

        return segment


class ChordLabelRenderer:
    def __init__(self,
                 text_anchor: Literal['center', 'start', 'end'] = 'center',
                 text_rotation_mode: Literal['up', 'down', 'left', 'right', 'absolute', 'relative'] | None = None,
                 text_rotation: float | Callable = 0.0,
                 text_displacement: Literal['auto'] | float = 0.0,
                 fixed_width: None | float = None,
                 accumulate_width: bool = False,
                 dpi: int = 100,
                 **kwargs):

        if text_rotation_mode == 'up':
            self.get_text_angle = lambda a: (a, -np.pi/2)
            self.va             = kwargs.pop('va',            'bottom')
            self.ha             = kwargs.pop('ha',            'center')
            self.rmode          = kwargs.pop('rotation_mode', 'anchor')

        elif text_rotation_mode == 'down':
            self.get_text_angle = lambda a: (a, +np.pi/2)
            self.va             = kwargs.pop('va',            'top')
            self.ha             = kwargs.pop('ha',            'center')
            self.rmode          = kwargs.pop('rotation_mode', 'anchor')

        elif text_rotation_mode == 'left':
            self.get_text_angle = lambda a: (a, 0.0)
            self.va             = kwargs.pop('va',            'center')
            self.ha             = kwargs.pop('ha',            'left')
            self.rmode          = kwargs.pop('rotation_mode', 'anchor')

        elif text_rotation_mode == 'right':
            self.get_text_angle = lambda a: (a, np.pi)
            self.va             = kwargs.pop('va',            'center')
            self.ha             = kwargs.pop('ha',            'right')
            self.rmode          = kwargs.pop('rotation_mode', 'anchor')

        elif text_rotation_mode == 'absolute':
            self.get_text_angle = lambda _, text_rotation = text_rotation : (0.0, text_rotation)
            self.va             = kwargs.pop('va',            'center')
            self.ha             = kwargs.pop('ha',            'center')
            self.rmode          = kwargs.pop('rotation_mode', 'anchor')

        elif text_rotation_mode == 'relative':
            self.get_text_angle = lambda a, text_rotation = text_rotation : (a, -text_rotation)
            self.va             = kwargs.pop('va',            'center')
            self.ha             = kwargs.pop('ha',            'center')
            self.rmode          = kwargs.pop('rotation_mode', 'anchor')

        else:
            self.get_text_angle = lambda _: (0.0, 0.0)
            self.va             = kwargs.pop('va',            'center')
            self.ha             = kwargs.pop('ha',            'center')
            self.rmode          = kwargs.pop('rotation_mode', 'anchor')

        if text_anchor == 'center':
            self.get_angle = lambda segment: segment['center_angle']
        elif text_anchor == 'start':
            self.get_angle = lambda segment: segment['start_angle']
        elif text_anchor == 'end':
            self.get_angle = lambda segment: segment['end_angle']

        self.text_displacement = text_displacement
        self.fixed_width = fixed_width
        self.accumulate_width = accumulate_width
        self.dpi = dpi
        self.kwargs = kwargs

    def __call__(self, ax, segment, **kwargs):
        _kwargs  = self.kwargs | kwargs
        r        = segment['radius'] + segment['width']
        center   = segment['center']
        label    = segment['label']
        angle    = self.get_angle(segment)
        text_rot = self.get_text_angle(angle)
        va       = _kwargs.pop('va', self.va)
        ha       = _kwargs.pop('ha', self.ha)
        rmode    = _kwargs.pop('rotation_mode', self.rmode)

        if self.text_displacement == 'auto':
            displacement = 0.002
        elif callable(self.text_displacement):
            displacement = self.text_displacement(segment, **kwargs)
        else:
            displacement = self.text_displacement

        x_label = center[0] + (r + displacement) * np.cos(angle)
        y_label = center[1] + (r + displacement) * np.sin(angle)

        # we rotate the text first only relative, so that we can exploit the
        # bbox to get the (real) height of the text-annulus (due to missing
        # curvature of the text, this will not be simply bbox.width) if the
        # width should be accumulated and no fixed-width is passed in
        textobj  = ax.text(0, 0, label, va=va, ha=ha,
                           rotation=np.rad2deg(text_rot[1]),
                           rotation_mode=rmode,
                           **_kwargs)

        accum = segment.get('width', 0.0)
        if self.accumulate_width:
            if self.fixed_width and isinstance(self.fixed_width, float):
                accum += self.fixed_width
            else:
                bbox = textobj.get_window_extent()
                # work in data coordinates
                transf = ax.transData.inverted()
                bbox = bbox.transformed(transf)

                # to get the height, we look at the bounding box and which point is the
                # furthest away, because we don't want to manipulate the rotation
                # itself. we then only take into consideration that which is beyond
                # 'width' of the segment. We could directly update to the value we
                # obtain, but keep the calculation for future changes/extensions
                corners = np.array([
                    [center[0] + bbox.x0 + bbox.width/2, center[1] + bbox.y0],
                    [center[0] + bbox.x0 + bbox.width/2, center[1] + bbox.y1],
                    [center[0] + bbox.x1 + bbox.width/2, center[1] + bbox.y0],
                    [center[0] + bbox.x1 + bbox.width/2, center[1] + bbox.y1],
                ])
                dist = np.max(np.sqrt((corners[:, 0] - center[0])**2 + (corners[:, 1] - center[1])**2))
                accum += dist

        # finally fully rotate the text
        textobj.set_position((x_label, y_label))
        textobj.set_rotation(np.rad2deg(text_rot[0] + text_rot[1]))
        segment['width'] = accum

        segment['ticklabel'] = textobj
        segment['ticklabel_kwargs'] = self.kwargs

        return segment


class ChordArcFlowRenderer:
    def __init__(self, displacement:float = 0.003, facecolor=None, **kwargs):
        self.displacement = displacement
        self.facecolor = facecolor
        self.kwargs = kwargs

    def __call__(self, ax, segment0, segment1, **kwargs):

        _kwargs = self.kwargs | kwargs

        theta1   = segment0['start_angle']
        theta2   = segment0['end_angle']
        radius_a = segment0['radius']

        theta3   = segment1['start_angle']
        theta4   = segment1['end_angle']
        radius_b = segment1['radius']

        fc       = self.facecolor if self.facecolor is not None else segment0['color']
        bezier_flow = ChordBezierFlow(theta1, theta2, radius_a-self.displacement,
                                     theta3, theta4, radius_b-self.displacement,
                                     facecolor=fc, **_kwargs)
        ax.add_patch(bezier_flow)

