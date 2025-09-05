# import cv2
import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.font_manager import findfont
# from .datamatrix.datamatrix import DataMatrix
from .data import ASSETS_DIR
from matal.utils import Bunch
import treepoem



FONT_SANSSERIF = 'Helvetica'

FONT_MONOSPACE_NARROW = ASSETS_DIR / 'fonts' / 'Iosevka' / 'iosevka-custom-slab-regular.ttf'
FONT_MONOSPACE_NARROW_BOLD = ASSETS_DIR / 'fonts' / 'Iosevka' / 'iosevka-custom-slab-medium.ttf'

FONT_MONOSPACE = ASSETS_DIR / 'fonts' / 'InconsolataLGC' / 'Inconsolata-LGC.ttf'
FONT_MONOSPACE_BOLD = ASSETS_DIR / 'fonts' / 'InconsolataLGC' / 'Inconsolata-LGC-Bold.ttf'

FONT_OCR = ASSETS_DIR / 'fonts' / 'ocr' / 'OCRB.ttf'

try:
    findfont(FONT_SANSSERIF, fallback_to_default=False)
except ValueError:
    FONT_SANSSERIF = 'Arial'


LABEL_TEMPLATES = {
    'avery_5267': Bunch(
        PAGE_W=8.5,
        PAGE_H=11,
        N_COLS=4,
        N_ROWS=20,
        CELL_W=1.75,
        CELL_H=0.5,
        LEFT=0.3,
        TOP=0.5,
        W_SPACE=0.3,
        H_SPACE=0,
        PA_TOP=0.825,
        PA_BOT=0.175,
        PA_FONTSIZE_S=5.5,
        PA_FONTSIZE_XS=4.5,
        PA_FONTSIZE_M=6,
        PA_FONTSIZE_L=7,
        PA_FONTSIZE_XL=9,
        PA_FONTSIZE_XXL=10,
        ),
    'avery_8195': Bunch(
        PAGE_W=8.5,
        PAGE_H=11,
        N_COLS=4,
        N_ROWS=15,
        CELL_W=1.75,
        CELL_H=2 / 3,
        LEFT=0.3,
        TOP=0.55,
        W_SPACE=0.3,
        H_SPACE=0,
        PA_TOP=0.825,
        PA_BOT=0.175,
        PA_FONTSIZE_S=5.5,
        PA_FONTSIZE_XS=4.5,
        PA_FONTSIZE_M=6,
        PA_FONTSIZE_L=7,
        PA_FONTSIZE_XL=9,
        PA_FONTSIZE_XXL=10,
        ),
    'avery_8460': Bunch(
        PAGE_W=8.5,
        PAGE_H=11,
        N_COLS=3,
        N_ROWS=10,
        CELL_W=2.625,
        CELL_H=1,
        LEFT=0.19,
        TOP=0.5,
        W_SPACE=0.12,
        H_SPACE=0,
        PA_TOP=0.825,
        PA_BOT=0.075,
        PA_FONTSIZE_S=7.0,
        PA_FONTSIZE_XS=6.5,

        ),
    'avery_8460_half': Bunch(
        PAGE_W=8.5,
        PAGE_H=11,
        N_COLS=3,
        N_ROWS=20,
        CELL_W=2.625,
        CELL_H=1/2,
        LEFT=0.19,
        TOP=0.5,
        W_SPACE=0.12,
        H_SPACE=0,
        PA_TOP=0.825,
        PA_BOT=0.075,
        PA_FONTSIZE_S=5.5,
        PA_FONTSIZE_XS=4.5,
        PA_FONTSIZE_M=6,
        PA_FONTSIZE_L=7,
        PA_FONTSIZE_XL=9,
        PA_FONTSIZE_XXL=10,
        ),
}

TEMPLATE_ALIAS = {
    'avery_8195': [f'avery_{i}' for i in
                   (15695, 18195, 18294, 18695, 38667, 42895, 48335, 5155, 5195, 6430, 6520, 6523, 6524, 88695)],
    'avery_5267': [f'avery_{i}' for i in
                   (15667, 18167, 18667, 48267, 48467, 48867, 5167, 5667, 5967, 8167, 8667, 8867, 8927, 95667)],
    'avery_8460': [f'avery_{i}' for i in
                   (15660, 15700, 15960, 16460, 16790, 18160, 18260, 18660, 22837, 28660, 32660, 38260, 45160, 48160, 48260, 48360, 48460, 48860, 48960, 5136, 5160, 5260, 55160, 5520, 55360, 5620, 5630, 5660, 58160, 58660, 5960, 6240, 6521, 6525, 6526, 6585, 80509, 8160, 8215, 8250, 85560, 8620, 8660, 88560, 8860, 8920, 95520, 95915)],
}

for k, v in TEMPLATE_ALIAS.items():
    for vi in v:
        LABEL_TEMPLATES[vi] = LABEL_TEMPLATES[k]


def load_kwargs(kwargs, color='black'):
    d = dict(color=kwargs.get('color', color),
             owner=kwargs.get('owner', 'SH'),
             sid=kwargs.get('sid', None),
             slot=kwargs.get('slot', None),
             well=kwargs.get('well', None),
             ot2_proc_sid=kwargs.get('ot2_proc_sid', None),
             src_vol=kwargs.get('src_vol', None),
             src_con=kwargs.get('src_con', None),
             batch_sid=kwargs.get('batch_sid', None),
             batch_i=kwargs.get('batch_i', None),
             mass=kwargs.get('mass', None),
             vol=kwargs.get('vol', None),
             method=kwargs.get('method', None),
             text=kwargs.get('text', None),
             part_i=kwargs.get('part_i', 1),
             n_parts=kwargs.get('n_parts', 1),
             additional_flags=kwargs.get('additional_flags', ''),
             barcode_only=kwargs.get('barcode_only', False),
             )
    for i in range(1, 11):
        d[f'src{i}_sid'] = kwargs.get(f'src{i}_sid', None)
        d[f'src{i}_frc'] = kwargs.get(f'src{i}_frc', None)
        d[f'src{i}_vol'] = kwargs.get(f'src{i}_vol', None)
    return Bunch(**d)


class RescaleFilter:
    def __init__(self, xscale=1.0, yscale=1.0):
        self.xscale = xscale
        self.yscale = yscale

    def __call__(self, im, dpi):
        ih, iw, c = im.shape

        fh = int(self.yscale * ih)
        fw = int(self.xscale * iw)

        im2 = cv2.resize(im, dsize=(fw, fh), interpolation=cv2.INTER_AREA)
        return im2, iw - fw, ih - fh


class LabelMaker:
    def __init__(self, tmpl='avery_5267'):
        self.c = LABEL_TEMPLATES[tmpl]

    @property
    def n_labels_per_page(self):
        return self.c.N_COLS * self.c.N_ROWS

    @property
    def n_labels_per_column(self):
        return self.c.N_ROWS
    
    _DRAW_FUNC_PREFIX = 'draw_'
    _DRAW_FUNC_SUFFIX = '_label'
    
    @property
    def available_label_styles(self):
        all_handles = [h for h in dir(self) if h.startswith(self._DRAW_FUNC_PREFIX) \
                       and h.endswith(self._DRAW_FUNC_SUFFIX) \
                       and (len(h) > len(self._DRAW_FUNC_PREFIX + self._DRAW_FUNC_SUFFIX)) \
                       and callable(getattr(self, h))]
        all_styles = [h[len(self._DRAW_FUNC_PREFIX):-len(self._DRAW_FUNC_SUFFIX)] for h in all_handles]
        return all_styles

    def create_axes(self, fig, border_width=0.0):
        axes = []
        for i in range(self.c.N_COLS):
            for j in range(self.c.N_ROWS):
                x0 = self.c.LEFT + i * (self.c.CELL_W + self.c.W_SPACE)
                y0 = self.c.TOP + j * (self.c.CELL_H + self.c.H_SPACE) + self.c.CELL_H
                ax = plt.Axes(fig, [x0 / self.c.PAGE_W, 1 - y0 / self.c.PAGE_H, self.c.CELL_W / self.c.PAGE_W,
                                    self.c.CELL_H / self.c.PAGE_H])
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                ax.set_xlim(0, self.c.CELL_W)
                ax.set_ylim(0, self.c.CELL_H)
                ax.axis('off')
                bb = mtransforms.Bbox([[0, 0], [self.c.CELL_W, self.c.CELL_H]])
                p_fancy = FancyBboxPatch((bb.xmin, bb.ymin), bb.width, bb.height,
                                         boxstyle='round,pad=0.,rounding_size=0.06',
#                                          boxstyle='round,pad=0.,rounding_size=0.00',
                                         ec='grey', fc='white', linestyle='-', lw=border_width)
                ax.add_patch(p_fancy)
                fig.add_axes(ax)
                axes.append(ax)
        return axes

    def _draw_owner(self, ax: matplotlib.axes.Axes, k: Bunch, x_loc=0.1, y_loc=0.7):
        if k.owner:
            ax.text(x_loc * self.c.CELL_W, y_loc * self.c.CELL_H, k.owner, color='white', font=FONT_MONOSPACE_NARROW_BOLD, fontsize=10,
                    horizontalalignment='center', verticalalignment='center', 
                    bbox=dict(facecolor=k.color, edgecolor='white', boxstyle='Round', alpha=1))

    def _draw_sid(self, ax: matplotlib.axes.Axes, k: Bunch):
        if k.sid:
            ax.text(0.82 * self.c.CELL_W, 0.5 * self.c.CELL_H, k.sid[:4] + '\n' + k.sid[4:], color=k.color, 
                    font=FONT_OCR, fontsize=10, horizontalalignment='center', verticalalignment='center',
                    bbox=dict(facecolor='white', edgecolor=k.color, alpha=1))

    def _draw_part_box(self, ax: matplotlib.axes.Axes, k: Bunch, dx: float = 0.0):
        part_color = ['green', 'mediumblue', 'm'][k.part_i - 1]
        p_fancy = Rectangle(((0.58 + dx) * self.c.CELL_W, 0.115 * self.c.CELL_H), 0.1 * self.c.CELL_W,
                            0.77 * self.c.CELL_H, ec=part_color, fc=part_color,
                            lw=0)
        ax.add_patch(p_fancy)
        # FIXME: italic style in Menlo does not work in matplotlib (mpl does not support .ttc font collections.)
        ax.text((0.62 + dx) * self.c.CELL_W, 0.69 * self.c.CELL_H, f'{k.part_i}', color='white',
                font=FONT_MONOSPACE, fontsize=9, fontstyle='italic',
                horizontalalignment='center', verticalalignment='center')

        ax.add_line(Line2D([(0.605 + dx) * self.c.CELL_W, (0.655 + dx) * self.c.CELL_W],
                           [0.45 * self.c.CELL_H, 0.55 * self.c.CELL_H], color='white', lw=0.8))

        ax.text((0.64 + dx) * self.c.CELL_W, 0.26 * self.c.CELL_H, f'{k.n_parts}', color='white',
                font=FONT_MONOSPACE, fontsize=9, fontstyle='italic',
                horizontalalignment='center', verticalalignment='center')

    def draw_label(self, ax: matplotlib.axes.Axes, style: str, **kwargs):
        assert style in self.available_label_styles
        func_name = self._DRAW_FUNC_PREFIX + style + self._DRAW_FUNC_SUFFIX
        getattr(self, func_name)(ax, **kwargs)
    
    def draw_dst_tube_label(self, ax: matplotlib.axes.Axes, **kwargs):
        k = load_kwargs(kwargs, color='green')

        self._draw_owner(ax, k)

        if k.ot2_proc_sid:
            ax.text(0.1 * self.c.CELL_W, 0.275 * self.c.CELL_H, k.ot2_proc_sid[:4] + '\n' + k.ot2_proc_sid[4:],
                    color='black', font=FONT_MONOSPACE_NARROW, fontsize=6,
                    horizontalalignment='center', verticalalignment='center')

        if k.n_parts != 1:
            dx = 0.065
            self._draw_part_box(ax, k)
        else:
            dx = 0

        ax.text((0.31 - dx) * self.c.CELL_W, 0.62 * self.c.CELL_H, 'slot', color='grey', fontname=FONT_SANSSERIF,
                fontsize=8,
                horizontalalignment='center', verticalalignment='bottom')
        ax.text((0.31 - dx) * self.c.CELL_W, 0.56 * self.c.CELL_H, k.slot, color='black', font=FONT_MONOSPACE, fontsize=16,
                horizontalalignment='center', verticalalignment='top')

        ax.text((0.53 - dx) * self.c.CELL_W, 0.62 * self.c.CELL_H, 'well', color='grey', fontname=FONT_SANSSERIF,
                fontsize=8,
                horizontalalignment='center', verticalalignment='bottom')
        ax.text((0.53 - dx) * self.c.CELL_W, 0.56 * self.c.CELL_H, k.well, color='black', font=FONT_MONOSPACE, fontsize=16,
                horizontalalignment='center', verticalalignment='top')
        self._draw_sid(ax, k)

    def draw_dst_tube_sep_label(self, ax: matplotlib.axes.Axes, **kwargs):
        k = load_kwargs(kwargs, color='green')

        if k.owner:
            ax.text(0.6 * self.c.CELL_W, 0.7 * self.c.CELL_H, k.owner, color='white', font=FONT_MONOSPACE, fontsize=9,
                    horizontalalignment='center', verticalalignment='center',
                    bbox=dict(facecolor=k.color, edgecolor='white', boxstyle='Round', alpha=1))

        if k.ot2_proc_sid:
            ax.text(0.6 * self.c.CELL_W, 0.3 * self.c.CELL_H, k.ot2_proc_sid[:4] + '\n' + k.ot2_proc_sid[4:],
                    color='black', font=FONT_MONOSPACE_NARROW, fontsize=6,
                    horizontalalignment='center', verticalalignment='center')

        ax.text(0.38 * self.c.CELL_W, 0.62 * self.c.CELL_H, 'slot', color='grey', fontname=FONT_SANSSERIF, fontsize=8,
                horizontalalignment='center', verticalalignment='bottom')
        ax.text(0.38 * self.c.CELL_W, 0.56 * self.c.CELL_H, k.slot, color='black', font=FONT_MONOSPACE, fontsize=16,
                horizontalalignment='center', verticalalignment='top')

        ax.text(0.16 * self.c.CELL_W, 0.62 * self.c.CELL_H, 'well', color='grey', fontname=FONT_SANSSERIF, fontsize=8,
                horizontalalignment='center', verticalalignment='bottom')
        ax.text(0.16 * self.c.CELL_W, 0.56 * self.c.CELL_H, k.well, color='black', font=FONT_MONOSPACE, fontsize=16,
                horizontalalignment='center', verticalalignment='top')
        self._draw_sid(ax, k)

    def draw_src_tube_label(self, ax: matplotlib.axes.Axes, **kwargs):
        k = load_kwargs(kwargs, color='red')

        self._draw_owner(ax, k)

        if k.n_parts != 1:
            dx = 0.03
            self._draw_part_box(ax, k, dx=0.08)
        else:
            dx = 0

        if k.ot2_proc_sid:
            ax.text(0.1 * self.c.CELL_W, 0.3 * self.c.CELL_H, k.ot2_proc_sid[:4] + '\n' + k.ot2_proc_sid[4:],
                    color='black', font=FONT_MONOSPACE_NARROW, fontsize=6,
                    horizontalalignment='center', verticalalignment='center')
        if k.src_con and isinstance(k.src_con, str):
            ax.text(0.25 * self.c.CELL_W, 0.90 * self.c.CELL_H, k.sid, color='black', font=FONT_MONOSPACE, fontsize=9,
                    horizontalalignment='left', verticalalignment='top')
            ax.text(0.25 * self.c.CELL_W, 0.55 * self.c.CELL_H, f'({k.src_con})', color='grey',
                    fontname=FONT_SANSSERIF, fontsize=7,
                    horizontalalignment='left', verticalalignment='center')
            ax.text(0.25 * self.c.CELL_W, 0.10 * self.c.CELL_H, f'{k.src_vol / 1000.0:.02f} mL', color='black',
                    fontname=FONT_SANSSERIF, fontsize=9,
                    horizontalalignment='left', verticalalignment='bottom')
        else:
            ax.text(0.25 * self.c.CELL_W, 0.55 * self.c.CELL_H, k.sid, color='black', font=FONT_MONOSPACE, fontsize=10,
                    horizontalalignment='left', verticalalignment='bottom')
            ax.text(0.25 * self.c.CELL_W, 0.45 * self.c.CELL_H, f'{k.src_vol / 1000.0:.02f} mL', color='black',
                    fontname=FONT_SANSSERIF, fontsize=10,
                    horizontalalignment='left', verticalalignment='top')

        ax.text((0.8 + dx) * self.c.CELL_W, 0.55 * self.c.CELL_H, k.slot, color='grey', font=FONT_MONOSPACE, fontsize=10,
                horizontalalignment='center', verticalalignment='bottom')
        ax.text((0.8 + dx) * self.c.CELL_W, 0.45 * self.c.CELL_H, k.well, color='grey', font=FONT_MONOSPACE, fontsize=10,
                horizontalalignment='center', verticalalignment='top')
    
    def draw_dish_comp_label(self, ax: matplotlib.axes.Axes, **kwargs):
        k = load_kwargs(kwargs, 'blue')
        self._draw_owner(ax, k)

        if k.method:
            ax.text(0.1 * self.c.CELL_W, 0.4 * self.c.CELL_H, k.method, color='black', fontname=FONT_SANSSERIF, fontsize=6,
                    horizontalalignment='center', verticalalignment='center')
        if k.mass:
            ax.text(0.1 * self.c.CELL_W, 0.2 * self.c.CELL_H, f'{round(k.mass)}\nmg', color='black',
                    fontname=FONT_SANSSERIF, fontsize=6,
                    horizontalalignment='center', verticalalignment='center')
        elif k.vol:
            ax.text(0.1 * self.c.CELL_W, 0.2 * self.c.CELL_H, f'{int(k.vol / 1000)}mL', color='black',
                    fontname=FONT_SANSSERIF, fontsize=6,
                    horizontalalignment='center', verticalalignment='center')

        x1, x2, x3 = 0.18 * self.c.CELL_W, 0.375 * self.c.CELL_W, 0.57 * self.c.CELL_W
        x1, x2, x3 = 0.17 * self.c.CELL_W, 0.425 * self.c.CELL_W, 0.59 * self.c.CELL_W
        
        n_ys = 0
        for si in range(1, 10):
            if getattr(k, f'src{si}_sid') is not None:
                n_ys += 1

        if k.additional_flags:
            y_locs = np.linspace(self.c.PA_TOP, self.c.PA_BOT, n_ys + 1, endpoint=True) * self.c.CELL_H
            fontsize = self.c.PA_FONTSIZE_XS
            ax.text(x2, self.c.PA_BOT * self.c.CELL_H, k.additional_flags, color='black', 
                    font=FONT_MONOSPACE, fontsize=fontsize,
                    horizontalalignment='center', verticalalignment='center')
        else:
            y_locs = np.linspace(self.c.PA_TOP, self.c.PA_BOT, n_ys, endpoint=True) * self.c.CELL_H
            fontsize = self.c.PA_FONTSIZE_S

        # print(y_locs, n_ys)
        yi = 0
        for si in range(1, n_ys + 1):
            if getattr(k, f'src{si}_sid'):
                sid = getattr(k, f'src{si}_sid')
                frc = getattr(k, f'src{si}_frc')
                vol = getattr(k, f'src{si}_vol')

                ax.text(x1, y_locs[yi], f'{sid}:', color='darkgrey', font=FONT_MONOSPACE_BOLD, fontsize=fontsize,
                        horizontalalignment='left', verticalalignment='center')
                ax.text(x2, y_locs[yi], f'{frc * 100:6.1f}%', color='black', font=FONT_MONOSPACE, fontsize=fontsize,
                        horizontalalignment='center', verticalalignment='center')
                ax.text(x3, y_locs[yi], f'{round(vol):5d}μL', color='black', font=FONT_MONOSPACE, fontsize=fontsize,
                        horizontalalignment='center', verticalalignment='center')
                yi += 1

        self._draw_sid(ax, k)

    def draw_barcode_label(self, ax: matplotlib.axes.Axes, **kwargs):
        k = load_kwargs(kwargs, 'cornflowerblue')
        if not k.barcode_only:
            self._draw_owner(ax, k)
            self._draw_sid(ax, k)

        text = f'{k.sid},{k.ot2_proc_sid}'
        rect = True
        
        image = treepoem.generate_barcode(
            barcode_type="datamatrix",
            options=dict(version='18x18'),
            data=text,
        )
        matrix = 1 - np.array(image.convert("1"), dtype=int)
        # dm = DataMatrix(text, rect=rect)
        # matrix = np.array(dm.matrix)
        matrix = np.kron(matrix, np.ones((4, 4)))

        ly, lx = matrix.shape
        if rect is False:
            y_size = 0.33 * self.c.CELL_H
            x_size = y_size / ly * lx
        else:
            x_size = 0.195 * self.c.CELL_W
            y_size = x_size / lx * ly
        x_center, y_center = 0.43 * self.c.CELL_W, 0.5 * self.c.CELL_H

        ax.imshow(matrix, extent=[x_center - x_size, x_center + x_size,
                                  y_center - y_size, y_center + y_size],
                  interpolation='none', aspect='equal', zorder=999, cmap='Greys', )

    def draw_dish_comp_barcode_label(self, ax: matplotlib.axes.Axes, **kwargs):
        k = load_kwargs(kwargs, 'blue')
        self._draw_owner(ax, k)

        ax.text(0.4 * self.c.CELL_W, 0.93 * self.c.CELL_H, k.sid, color=k.color, font=FONT_MONOSPACE_BOLD, fontsize=6,
                horizontalalignment='center', verticalalignment='top')

        if k.method:
            ax.text(0.1 * self.c.CELL_W, 0.4 * self.c.CELL_H, k.method, color='black', font=FONT_SANSSERIF, fontsize=6,
                    horizontalalignment='center', verticalalignment='center')
        if k.mass:
            ax.text(0.1 * self.c.CELL_W, 0.2 * self.c.CELL_H, f'{int(k.mass * 1000)}mg', color='black',
                    font=FONT_SANSSERIF, fontsize=6,
                    horizontalalignment='center', verticalalignment='center')
        elif k.vol:
            ax.text(0.1 * self.c.CELL_W, 0.2 * self.c.CELL_H, f'{int(k.vol / 1000)}mL', color='black',
                    font=FONT_SANSSERIF, fontsize=6,
                    horizontalalignment='center', verticalalignment='center')

        x1, x2, x3 = 0.18 * self.c.CELL_W, 0.375 * self.c.CELL_W, 0.57 * self.c.CELL_W

        if k.additional_flags:
            y_locs = np.linspace(0.685, 0.105, 5, endpoint=True) * self.c.CELL_H
            fontsize = 5.5
            ax.text(x2, 0.18 * self.c.CELL_H, k.additional_flags, color='black', font=FONT_MONOSPACE, fontsize=fontsize,
                    horizontalalignment='center', verticalalignment='center')
        else:
            y_locs = np.linspace(0.685, 0.105, 5, endpoint=True) * self.c.CELL_H
            fontsize = 5.5

        yi = 0
        for si in range(1, 6):
            if getattr(k, f'src{si}_sid'):
                sid = getattr(k, f'src{si}_sid')
                frc = getattr(k, f'src{si}_frc')
                vol = getattr(k, f'src{si}_vol')

                ax.text(x1, y_locs[yi], f'{sid[:3]}:', color='darkgrey', font=FONT_MONOSPACE_BOLD, fontsize=fontsize,
                        horizontalalignment='left', verticalalignment='center')
                ax.text(x2, y_locs[yi], f'{frc * 100:6.1f}%', color='black', font=FONT_MONOSPACE, fontsize=fontsize,
                        horizontalalignment='center', verticalalignment='center')
                ax.text(x3, y_locs[yi], f'{round(vol):5d}μL', color='black', font=FONT_MONOSPACE, fontsize=fontsize,
                        horizontalalignment='center', verticalalignment='center')
                yi += 1

        text = f'{k.sid},{k.ot2_proc_sid}'
        rect = False
        # dm = DataMatrix(text, rect=rect)
        # matrix = np.array(dm.matrix)
        image = treepoem.generate_barcode(
            barcode_type="datamatrix",
            options=dict(version='18x18'),
            data=text,
        )
        matrix = 1 - np.array(image.convert("1"), dtype=int)
        # dm = DataMatrix(text, rect=rect)
        # matrix = np.array(dm.matrix)
        matrix = np.kron(matrix, np.ones((4, 4)))

        ly, lx = matrix.shape
        if rect is False:
            y_size = 0.35 * self.c.CELL_H
            x_size = y_size / ly * lx
        else:
            x_size = 0.195 * self.c.CELL_W
            y_size = x_size / lx * ly
        x_center, y_center = 0.83 * self.c.CELL_W, 0.5 * self.c.CELL_H

        ax.imshow(matrix, extent=[x_center - x_size, x_center + x_size,
                                  y_center - y_size, y_center + y_size],
                  interpolation='none', aspect='equal', zorder=999, cmap='Greys', )

    def draw_oven_boat_label(self, ax: matplotlib.axes.Axes, **kwargs):
        k = load_kwargs(kwargs, 'sienna')
        self._draw_owner(ax, k, x_loc=0.2, y_loc=0.2)

        ax.text(0.6 * self.c.CELL_W, 0.2 * self.c.CELL_H, k.sid, color=k.color, font=FONT_OCR, fontsize=9,
                horizontalalignment='center', verticalalignment='center',
                bbox=dict(facecolor='white', edgecolor=k.color, alpha=0))

    def draw_text_label(self, ax: matplotlib.axes.Axes, hightlight_firstline=True, **kwargs):
        k = load_kwargs(kwargs)
        
        if hightlight_firstline:
            lines = k.text.split('\n')

            hl_text = '\n'.join([l if i == 0 else '' for i, l in enumerate(lines)])
            nm_text = '\n'.join([l if i != 0 else '' for i, l in enumerate(lines)])
            
            ax.text(0.5 * self.c.CELL_W, 0.5 * self.c.CELL_H, hl_text, color='firebrick', font=FONT_MONOSPACE_BOLD, fontsize=6,
                horizontalalignment='center', verticalalignment='center')
            ax.text(0.5 * self.c.CELL_W, 0.5 * self.c.CELL_H, nm_text, color='black', font=FONT_MONOSPACE, fontsize=6,
                horizontalalignment='center', verticalalignment='center')
        else:
            ax.text(0.5 * self.c.CELL_W, 0.5 * self.c.CELL_H, k.text, color='black', font=FONT_MONOSPACE, fontsize=6,
                horizontalalignment='center', verticalalignment='center')