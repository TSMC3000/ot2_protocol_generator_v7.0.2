import numpy as np
import pandas as pd
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt


COLOR_GROUPS = [
    ['LAP', 'MMT', 'CMC', 'CNF', 'SLK', ],                                     # R: Nanosheets, fibers
    ['AGR', 'ALG', 'CAR', 'CHS', 'PUL', 'GEL', 'GLU', 'ZIN', ],                # G: Protein & large molecules
    ['GLY', 'FFA', 'LAC', 'LEV', 'PHA', 'SUA', 'XYL']                          # B: Crosslinkers & small molecules
]

def comp_to_colors(comp, color_groups=COLOR_GROUPS):
    # Convert compositions to colors based on material types
    colors = np.array([comp[cols].sum(axis=1) for cols in color_groups]).T
    return np.clip(colors, 0, 1) # Clip ranges within [0, 1] in case of any rounding errors

def add_comp_colorbars(ax, fraction=0.05, pad=0.1, shrink=0.8, color_groups=COLOR_GROUPS, material_name_lut=None):
    fig = ax.get_figure()

    # Extend the figure to fit colorbars
    fig_w, fig_h = fig.get_size_inches()
    new_fig_w = fig_w / (1 - fraction - pad)
    fig.set_size_inches(new_fig_w, fig_h)

    # Calculate bounding boxes for 3 colorbars
    # Some codes from https://github.com/matplotlib/matplotlib/blob/v3.6.0/lib/matplotlib/colorbar.py#L199-L1329
    parents_bbox = mpl.transforms.Bbox.union([ax.get_position(original=True).frozen(), ])
    pb1, _, pbcb = parents_bbox.splitx(1 - fraction - pad, 1 - fraction)
    shrinking_trans = mpl.transforms.BboxTransform(parents_bbox, pb1)
    new_posn = shrinking_trans.transform(ax.get_position(original=True))
    new_posn = mpl.transforms.Bbox(new_posn)
    ax.set_position(new_posn)

    # Split bbox into 3 since we need 3 colorbars (R, G, and B)
    pbcb_list = pbcb.splity(1/3, 2/3)


    cax_list = [fig.add_axes(pbcb_i.shrunk(1.0, shrink), 
                             label=f'<colorbar {i}>') for i, pbcb_i in enumerate(pbcb_list)]
    cax_list = reversed(cax_list)
    
    # Place new colorbars
    cbars = [
        fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(),
                                           cmap=mpl.colors.LinearSegmentedColormap.from_list(f'grad_{color}', ['black', color])),
                     cax=cax) for cax, color in zip(cax_list, ['red', 'green', 'blue'])
    ]

    # Display material names alongside colorbars
    for cbar, mat in zip(cbars, color_groups):
        cbar.set_ticks([0, 1])
        cbar.ax.set_ylabel('\n'.join([material_name_lut[m] if material_name_lut else m for m in mat]))


def _ma_proxy(f):
    def proxy_f(a):
        if np.ma.isMaskedArray(a):
            new_a = np.copy(a)
            new_a.data = f(a.data)
            return new_a
        else:
            return f(a)
    return proxy_f


def inverse_density_scale(scatters, ax, n_bins=51, smooth_coeff=0.005, apply=True, on='xyz', target_shape='square', scale=1.0):
    interpolate_kw = dict(fill_value='extrapolate', kind='linear')
    
    all_scatter_pts = np.array([item for sc in scatters for item in sc.get_offsets().data])
    
    on_axis = []
    for axis in 'xyz':
        if axis not in on: continue
        if not hasattr(ax, f'set_{axis}scale'): continue
        on_axis.append(axis)
    
    hists, edges = {}, {}
    for col, axis in enumerate(on_axis):
        hist, bin_edges = np.histogram(all_scatter_pts[:,col], bins=n_bins, density=True)

        hist /= hist.sum()
        hist += smooth_coeff
        hist /= hist.sum()
        
        hists[axis] = hist
        edges[axis] = bin_edges
        
    funcs = {}
    for col, axis in enumerate(on_axis):
        if target_shape == 'square':
            f = scipy.interpolate.interp1d(edges[axis][:-1], np.cumsum(hists[axis]) * scale, **interpolate_kw)
            fr = scipy.interpolate.interp1d(np.cumsum(hists[axis]) * scale, edges[axis][:-1], **interpolate_kw)
        elif target_shape == 'sphere':
            pass
        else:
            raise ValueError(f'Unknown target_shape: {target_shape}')
        
        funcs[axis] = (_ma_proxy(f), _ma_proxy(fr))
        if apply:
            getattr(ax, f'set_{axis}scale')('function', functions=funcs[axis])

    return funcs