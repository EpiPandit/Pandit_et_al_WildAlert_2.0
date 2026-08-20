# coding: utf-8
"""
WildAlert — State Case Density Map
===================================
Drop-in companion to plot_multiple_anomaly_timelines.

Usage
-----
from WildAlertpy import anomaly_models as an
import geopandas as gpd

state_shapes = an.prepare_us_state_shapes(
    'C:/path/to/cb_2018_us_state_500k.shp'
)

an.plot_state_case_density_map(
    filenames=[
        'C:/…/data/paper_exports/endemic/shorebirds-with-neurologic-disease-in-california',
    ],
    grey_periods=[
        ('2022-03-01', '2022-06-30'),
        ('2023-01-15', '2023-04-30'),
    ],
    disease_labels=['H5N1 outbreak', 'H5N1 outbreak'],   # same label → same colour
    state_shapes=state_shapes,
    title='Shorebirds with neurologic disease – California',
    map_style='hexbin',          # 'hexbin' | 'kde' | 'dot'
    save=True,
    save_path='shorebird_california_density.png',
)
"""

from __future__ import annotations

import os
from typing import Literal

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_and_geocode(filename: str):
    """Read an .xlsx case file and return a GeoDataFrame."""
    import geopandas as gpd
    from shapely.geometry import Point
    from WildAlertpy import read_data as read_data   # noqa: PLC0415

    df = read_data.read_data(filename=filename + ".xlsx")

    lon_col = next(
        (c for c in ['longitude_found', 'longitude', 'lon', 'long'] if c in df.columns),
        None
    )
    lat_col = next(
        (c for c in ['latitude_found', 'latitude', 'lat'] if c in df.columns),
        None
    )
    if lon_col is None or lat_col is None:
        raise KeyError(
            f"Could not find lat/lon columns in {filename}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.dropna(subset=[lon_col, lat_col])
    geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    date_col = next(
        (c for c in ['found_date', 'date_found', 'collection_date', 'date'] if c in gdf.columns),
        None
    )
    if date_col:
        gdf['_date'] = pd.to_datetime(gdf[date_col], errors='coerce')

    return gdf


def _assign_period_labels(
    gdf,
    grey_periods: list[tuple[str, str]] | None,
    disease_labels: list[str] | None,
    default_label: str = 'Background cases',
):
    """Add a 'period_label' column to gdf based on outbreak periods."""
    gdf = gdf.copy()
    gdf['period_label'] = default_label

    if grey_periods and disease_labels and '_date' in gdf.columns:
        for (start, end), label in zip(grey_periods, disease_labels):
            mask = (
                (gdf['_date'] >= pd.to_datetime(start)) &
                (gdf['_date'] <= pd.to_datetime(end))
            )
            gdf.loc[mask, 'period_label'] = label

    return gdf


def _build_label_colors(
    disease_labels: list[str] | None,
    default_label: str = 'Background cases',
    default_color: str = '#AAAAAA',
):
    """Return an ordered dict mapping label → hex colour."""
    from matplotlib.colors import to_hex

    unique = list(dict.fromkeys(disease_labels)) if disease_labels else []
    palette = cm.get_cmap('Paired', max(len(unique), 1))
    label_colors = {lab: to_hex(palette(i)) for i, lab in enumerate(unique)}
    label_colors[default_label] = default_color
    return label_colors


def _detect_state(save_path: str, state_shapes: dict) -> str | None:
    """Guess the target state from the save_path string."""
    for state in state_shapes:
        if state.lower() in save_path.lower():
            return state
    return None


# ---------------------------------------------------------------------------
# Drawing helpers – one per map_style
# ---------------------------------------------------------------------------

def _draw_hexbin(ax, xs, ys, extent, cmap='YlOrRd', gridsize=50, log_scale=False):
    """
    Hex-bin density layer.  Returns the hexbin collection for the colorbar.
    log_scale=True applies log2 colour scaling via LogNorm — useful when a
    few hotspot cells would otherwise wash out the rest of the map.
    """
    from matplotlib.colors import LogNorm, Normalize

    ax.set_facecolor('white')

    # Probe pass to get actual count range before applying norm
    hb_probe = ax.hexbin(
        xs, ys, gridsize=gridsize, cmap=cmap, mincnt=1,
        extent=extent, alpha=0.0, linewidths=0,
    )
    counts = hb_probe.get_array()
    hb_probe.remove()

    if log_scale and len(counts) > 0 and counts.max() > 1:
        norm = LogNorm(vmin=1, vmax=float(counts.max()))
    else:
        norm = Normalize(vmin=1, vmax=float(counts.max()) if len(counts) > 0 else 1)

    hb = ax.hexbin(
        xs, ys,
        gridsize=gridsize,
        cmap=cmap,
        mincnt=1,
        extent=extent,
        norm=norm,
        alpha=0.85,
        linewidths=0.6,
        edgecolors="#757575"     #'#CCCCCC',
    )
    hb._log_scale = log_scale
    return hb


def _draw_kde(ax, xs, ys, extent, color='darkorange', n_levels=8):
    """
    KDE contour filled layer using scipy.  Falls back to scatter if too few pts.
    """
    from scipy.stats import gaussian_kde

    if len(xs) < 5:
        ax.scatter(xs, ys, s=8, alpha=0.6, color=color)
        return None

    xmin, xmax, ymin, ymax = extent
    xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values    = np.vstack([xs, ys])
    kernel    = gaussian_kde(values)
    z         = np.reshape(kernel(positions).T, xx.shape)

    cmap_fade = plt.cm.get_cmap('YlOrRd')
    cf = ax.contourf(xx, yy, z, levels=n_levels, cmap=cmap_fade, alpha=0.75)
    ax.contour(xx, yy, z, levels=n_levels, colors='white', linewidths=0.4, alpha=0.5)
    return cf


def _draw_dot(ax, xs, ys, color, label, size=12, alpha=0.75):
    """
    Proportional dot scatter with mild jitter so overlapping points spread out.
    """
    rng = np.random.default_rng(42)
    spread = 0.03   # degrees – tiny geographic jitter
    jx = xs + rng.uniform(-spread, spread, len(xs))
    jy = ys + rng.uniform(-spread, spread, len(ys))
    ax.scatter(jx, jy, s=size, color='red', alpha=alpha, label=label,
               edgecolors='white', linewidths=1.2, zorder=3)
 


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def plot_state_case_density_map(
    filenames: list[str],
    state_shapes: dict,
    grey_periods: list[tuple[str, str]] | None = None,
    disease_labels: list[str] | None = None,
    title: str | None = None,
    map_style: Literal['hexbin', 'kde', 'dot'] = 'hexbin',
    cmap: str = 'YlOrRd',
    hexbin_gridsize: int = 50,
    log_scale: bool = False,
    figsize: tuple[float, float] = (8.27, 11.0),   # A4 portrait
    basemap_color: str = '#F5F5F0',
    county_color: str = '#DDDDCC',
    border_color: str = '#555555',
    show_colorbar: bool = True,
    show_legend: bool = True,
    save: bool = False,
    save_path: str | None = None,
    dpi: int = 300,
):
    """
    Plot a state-level case density map for one or more WildAlert data files.

    Parameters
    ----------
    filenames       : list of full paths (without .xlsx extension), same as
                      used in plot_multiple_anomaly_timelines.
    state_shapes    : dict returned by prepare_us_state_shapes().
    grey_periods    : list of (start_date, end_date) outbreak windows.
    disease_labels  : list of labels matching grey_periods.
    title           : map title (auto-generated if None).
    map_style       : 'hexbin'  – colour-intensity hex grid (best for many pts)
                      'kde'     – smooth kernel-density contours (best for
                                  visualising hotspot shape)
                      'dot'     – coloured dots per outbreak period with jitter
                                  (best for small N or comparing periods)
    cmap            : matplotlib colormap name used by hexbin / kde styles.
    hexbin_gridsize : number of hexagons along each axis (hexbin only).
    log_scale       : apply log2 colour scaling to hexbin. Useful when a few
                      dense hotspots would otherwise wash out low-density areas.
                      Colorbar ticks are relabelled as powers of 2 (1, 2, 4 …).
                      Ignored when map_style is 'kde' or 'dot'.
    figsize         : (width_inches, height_inches).
    basemap_color   : fill for state polygon.
    county_color    : fill for county subdivisions (if available).
    border_color    : state outline colour.
    show_colorbar   : show density colour scale (hexbin / kde only).
    show_legend     : show outbreak-period legend.
    save            : write PNG to disk.
    save_path       : output path; auto-generated if None.
    dpi             : resolution.

    Returns
    -------
    fig, ax
    """
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    # ------------------------------------------------------------------
    # 1. Detect target state
    # ------------------------------------------------------------------
    found_state = None
    if save_path:
        found_state = _detect_state(save_path, state_shapes)
    if found_state is None and filenames:
        found_state = _detect_state(filenames[0], state_shapes)
    if found_state is None:
        # Fall back to first state in shapes dict that isn't 'USA'
        candidates = [s for s in state_shapes if s != 'USA']
        found_state = candidates[0] if candidates else None
    if found_state is None:
        raise ValueError("Could not determine target state from filenames or save_path.")

    state_gdf = state_shapes[found_state].to_crs("EPSG:4326")

    # ------------------------------------------------------------------
    # 2. Build colour mapping before loading data
    # ------------------------------------------------------------------
    default_label  = 'Background cases'
    label_colors   = _build_label_colors(disease_labels, default_label)

    # ------------------------------------------------------------------
    # 3. Load, geocode, clip, and label all files
    # ------------------------------------------------------------------
    all_gdfs = []
    for fn in filenames:
        try:
            gdf = _load_and_geocode(fn)
            gdf = gdf[gdf.within(state_gdf.unary_union)]
            if gdf.empty:
                print(f"[density map] No points inside {found_state} for {fn}")
                continue
            gdf = _assign_period_labels(gdf, grey_periods, disease_labels, default_label)
            all_gdfs.append(gdf)
        except Exception as exc:
            print(f"[density map] Skipping {fn}: {exc}")

    if not all_gdfs:
        print("[density map] No valid data to plot.")
        return None, None

    combined = pd.concat(all_gdfs, ignore_index=True)
    combined_gdf = gpd.GeoDataFrame(combined, geometry='geometry', crs="EPSG:4326")

    # ------------------------------------------------------------------
    # 4. Figure layout  (no cartopy / geoplot required – plain matplotlib)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    ax.axis('off')

    # -- state boundary fill
    state_gdf.plot(ax=ax, facecolor=basemap_color, edgecolor=border_color,
                   linewidth=1.2, zorder=1)

    # -- bounding box with a small padding
    bounds = state_gdf.total_bounds       # xmin, ymin, xmax, ymax
    pad_x  = (bounds[2] - bounds[0]) * 0.02
    pad_y  = (bounds[3] - bounds[1]) * 0.02
    ax.set_xlim(bounds[0] - pad_x, bounds[2] + pad_x)
    ax.set_ylim(bounds[1] - pad_y, bounds[3] + pad_y)
    extent = (bounds[0] - pad_x, bounds[2] + pad_x,
              bounds[1] - pad_y, bounds[3] + pad_y)

    # ------------------------------------------------------------------
    # 5. Data layers
    # ------------------------------------------------------------------
    colorbar_obj = None

    if map_style == 'hexbin':
        # ── single hexbin across ALL points for overall density
        xs = combined_gdf.geometry.x.values
        ys = combined_gdf.geometry.y.values
        hb = _draw_hexbin(ax, xs, ys, extent, cmap=cmap,
                          gridsize=hexbin_gridsize, log_scale=log_scale)
        if show_colorbar:
            import matplotlib.ticker as ticker
            cb = fig.colorbar(hb, ax=ax, shrink=0.45, pad=0.02)
            if log_scale:
                cb.set_label('Cases per hex cell (log\u2082 scale)', size=9)
                # Re-label ticks as powers of 2: 1, 2, 4, 8, 16 …
                from matplotlib.colors import LogNorm
                import numpy as np
                vmax = hb.get_array().max()
                powers = [2**i for i in range(0, int(np.log2(max(vmax, 1))) + 1)]
                cb.set_ticks(powers)
                cb.set_ticklabels([str(p) for p in powers])
            else:
                cb.set_label('Cases per hex cell', size=9)
            cb.ax.tick_params(labelsize=9)
            colorbar_obj = cb

        # ── overlay outbreak-period scatter on top
        if grey_periods and '_date' in combined_gdf.columns:
            for (start, end), label in zip(grey_periods, disease_labels):
                subset = combined_gdf[
                    (combined_gdf['_date'] >= pd.to_datetime(start)) &
                    (combined_gdf['_date'] <= pd.to_datetime(end))
                ]
                if subset.empty:
                    continue
                ax.scatter(
                    subset.geometry.x, subset.geometry.y,
                    s=14, color=label_colors.get(label, '#FF0000'),
                    edgecolors='white', linewidths=0.5,
                    alpha=0.85, zorder=5, label=label,
                )

    elif map_style == 'kde':
        xs = combined_gdf.geometry.x.values
        ys = combined_gdf.geometry.y.values
        cf = _draw_kde(ax, xs, ys, extent, n_levels=9)
        if cf is not None and show_colorbar:
            cb = fig.colorbar(cf, ax=ax, shrink=0.45, pad=0.02,
                              label='Relative case density')
            cb.ax.tick_params(labelsize=9)

        # outbreak dots over the KDE
        if grey_periods and '_date' in combined_gdf.columns:
            for (start, end), label in zip(grey_periods, disease_labels):
                subset = combined_gdf[
                    (combined_gdf['_date'] >= pd.to_datetime(start)) &
                    (combined_gdf['_date'] <= pd.to_datetime(end))
                ]
                if subset.empty:
                    continue
                _draw_dot(ax, subset.geometry.x.values,
                          subset.geometry.y.values,
                          color=label_colors.get(label, '#FF0000'),
                          label=label, size=40, alpha=0.85)

    elif map_style == 'dot':
        # All unique labels in draw order: background first, outbreaks on top
        ordered_labels = [default_label] + [
            l for l in (list(dict.fromkeys(disease_labels))
                        if disease_labels else [])
        ]
        for label in ordered_labels:
            subset = combined_gdf[combined_gdf['period_label'] == label]
            if subset.empty:
                continue
            size   = 18 if label == default_label else 40
            alpha  = 0.35 if label == default_label else 0.85
            _draw_dot(ax, subset.geometry.x.values,
                      subset.geometry.y.values,
                      color=label_colors.get(label, '#AAAAAA'),
                      label=label, size=size, alpha=alpha)

    else:
        raise ValueError(f"map_style must be 'hexbin', 'kde', or 'dot'. Got: {map_style!r}")

    # -- state outline drawn again on top so it isn't buried under data
    state_gdf.plot(ax=ax, facecolor='none', edgecolor=border_color,
                   linewidth=1.4, zorder=10)

    # ------------------------------------------------------------------
    # 6. Legend
    # ------------------------------------------------------------------
    if show_legend:
        legend_handles = []

        if map_style == 'dot':
            # Dot legend – all period labels + background
            for label, color in label_colors.items():
                s    = 6  if label == default_label else 14
                alph = 0.45 if label == default_label else 0.85
                legend_handles.append(
                    plt.scatter([], [], s=s, color=color,
                                alpha=alph, label=label)
                )
        elif grey_periods:
            # Hexbin / KDE: coloured dot per outbreak period only
            seen = set()
            for (start, end), label in zip(grey_periods, disease_labels):
                if label in seen:
                    continue
                seen.add(label)
                legend_handles.append(
                    plt.scatter([], [], s=50,
                                color=label_colors.get(label, '#FF0000'),
                                edgecolors='white', linewidths=0.5,
                                label=label)
                )

        if legend_handles:
            leg = ax.legend(
                handles=legend_handles,
                loc='lower left',
                fontsize=9,
                framealpha=0.90,
                edgecolor='#CCCCCC',
                title='Outbreak periods',
                title_fontsize=9,
            )
            leg.get_frame().set_linewidth(0.5)

    # ------------------------------------------------------------------
    # 7. Title, stats annotation, styling
    # ------------------------------------------------------------------
    n_total    = len(combined_gdf)
    n_outbreak = (
        (combined_gdf['period_label'] != default_label).sum()
        if 'period_label' in combined_gdf.columns
        else 0
    )

    map_title = title or (
        os.path.basename(filenames[0]).replace("-", " ").title()
        if filenames else "Case density map"
    )
    ax.set_title(
        map_title,
        fontsize=13,
        fontweight='bold',
        pad=12,
        loc='center',
    )

    # Small annotation box – total/outbreak counts + style note
    style_note = {
        'hexbin': 'Hex-bin density',
        'kde':    'KDE contours',
        'dot':    'Jittered dot plot',
    }[map_style]
    annotation = (
        f"{style_note}  ·  n = {n_total:,} geocoded cases"
        + (f"  ·  {n_outbreak:,} during outbreak periods" if grey_periods else "")
        + f"\n{found_state}"
    )
    ax.annotate(
        annotation,
        xy=(0.5, 0.005),
        xycoords='axes fraction',
        fontsize=8,
        color='#666666',
        ha='center',
        va='bottom',
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()

    # ------------------------------------------------------------------
    # 8. Save
    # ------------------------------------------------------------------
    if save:
        out_path = save_path
        if out_path is None:
            slug = map_title.replace(' ', '_').lower()[:50]
            out_path = f"{slug}_{found_state}_{map_style}_map.png"
        plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
        print(f"[density map] Saved to: {out_path}")

    plt.show()
    return fig, ax