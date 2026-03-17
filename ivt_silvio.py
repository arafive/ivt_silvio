"""daniele.carnevale, data di creazione: Thu Oct 31 12:35:15 2024."""

import os

import numpy as np
import pandas as pd
import xarray as xr

import cartopy.crs as ccrs
import cartopy.feature as cf

import matplotlib.pyplot as plt

from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
import matplotlib.ticker as mticker

### Installare --> pip install netCDF4

lista_possibili_cartelle_lavoro = [
    '/media/daniele/Daniele2TB/repo/ivt_silvio',
    '/run/media/daniele.carnevale/Daniele2TB/repo/ivt_silvio',
    ]

cartella_lavoro = [x for x in lista_possibili_cartelle_lavoro if os.path.exists(x)][0]

cartella_lavoro 
os.chdir(cartella_lavoro)

# %% PARAMETRI GENERALI

g                   = 9.80665
livelli_plot_ivt    = [250, 300, 400, 500, 600, 700, 800, 900, 1000]
skip_frecce         = 5
cartella_file_nc    = '/run/media/daniele.carnevale/Daniele2TB/test/ivt_silvio/gfs'
nome_dataset_finale = 'IVTxy_dataset_completo'
cartella_output_png = '/run/media/daniele.carnevale/Daniele2TB/test/ivt_silvio/figure'

os.makedirs(cartella_output_png, exist_ok=True)

# # # # # # # #
# # # # # # # #


lista_file_nc = sorted(os.listdir(cartella_file_nc))

print(f'\n{lista_file_nc}\n')

lista_ivtx = []
lista_ivty = []
lista_tempi = []
lista_ivt = []

for file_nc in lista_file_nc:
    ds = xr.open_dataset(f'{cartella_file_nc}/{file_nc}', engine='netcdf4')
    
    inizio_run = pd.to_datetime(ds['time'].values)
    str_inizio_run = str(inizio_run[0]).replace(' ', '_').replace(':', '-')
    livelli = ds['plev'].values.astype(int)[::-1] # Pa
    lat = ds['lat'].values
    lon = ds['lon'].values
    lon_2D, lat_2D = np.meshgrid(lon, lat)
    
    q = ds['q'].values # kg kg**-1
    u = ds['u'].values # m s**-1
    v = ds['v'].values # m s**-1
    
    ### Prendo solo l'unico tempo
    q = q[0, ...]
    u = u[0, ...]
    v = v[0, ...]
    
    # # # # # # # #
    # # # # # # # #
    
    IVTx = np.zeros(shape=q.shape[1:])
    IVTy = np.zeros(shape=q.shape[1:])
    
    for (ind_livello, liv1), liv2 in zip(enumerate(livelli), livelli[1:]):
        
        dp = liv1 - liv2
        
        IVTx = IVTx + u[ind_livello, ...] * q[ind_livello, ...] * dp / g
        IVTy = IVTy + v[ind_livello, ...] * q[ind_livello, ...] * dp / g
        
    IVT = np.sqrt(IVTx ** 2 + IVTy ** 2)
    
    lista_ivtx.append(IVTx)
    lista_ivty.append(IVTy)
    lista_ivt.append(IVT)
    lista_tempi.append(str(pd.to_datetime(ds['time'].values[0])))

# # # # # # # #
# # # # # # # #

stack_ivtx = np.stack(lista_ivtx, axis=0)
stack_ivty = np.stack(lista_ivty, axis=0)

ds_ivtxy = xr.Dataset(
    {'ivtx': (('time', 'lat', 'lon'), stack_ivtx),
     'ivty': (('time', 'lat', 'lon'), stack_ivty)},
    
    coords={
        'time': lista_tempi,
        'lon': ds['lon'].values,
        'lat': ds['lat'].values
    }
)

ds_ivtxy.to_netcdf(f'{cartella_lavoro}/{nome_dataset_finale}.nc')

# %%

for ivt, ivtx, ivty, t in zip(lista_ivt, lista_ivtx, lista_ivty, lista_tempi):
    nome_png_output = t.replace(' ', '_').replace(':', '-')
    
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent((lon_2D.min(), lon_2D.max(), lat_2D.min(), 57))
    ax.add_feature(cf.COASTLINE, lw=1, edgecolor='black')
    
    ivt_plot = ax.contourf(
        lon_2D,
        lat_2D,
        ivt,
        transform=ccrs.PlateCarree(),
        levels=livelli_plot_ivt,
        cmap='coolwarm',
        extend='max'
    )
    
    frecce_plot = ax.quiver(
        lon_2D[::skip_frecce, ::skip_frecce],
        lat_2D[::skip_frecce, ::skip_frecce],
        ivtx[::skip_frecce, ::skip_frecce],
        ivty[::skip_frecce, ::skip_frecce],
        scale=15_000
    )
    
    cbar = plt.colorbar(ivt_plot)
    cbar.set_ticks(livelli_plot_ivt)
    
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=1,
        color='gray',
        alpha=0.5,
        linestyle='--')
    
    gl.top_labels = False
    gl.right_labels = False
    gl.ylabel_style = {'size': 7, 'color': 'black', 'weight': 'normal'}
    gl.xlabel_style = {'size': 7, 'color': 'black', 'weight': 'normal'}
    
    gl.xlocator = mticker.FixedLocator(range(int(lon_2D.min()), int(lon_2D.max()) + 1, 5))
    gl.xformatter = LongitudeFormatter(degree_symbol="°")
    
    gl.ylocator = mticker.FixedLocator(range(int(lat_2D.min()), int(lat_2D.max()) + 1, 5))
    gl.yformatter = LatitudeFormatter(degree_symbol="°")
    
    ax.set_aspect('auto', adjustable=None)
    plt.title(f'Massimo: {int(np.max(ivt))} kg/m*s')
    plt.savefig(f"{cartella_output_png}/{nome_png_output}.png", dpi=300, format='png', bbox_inches='tight')
    
    plt.show()
    plt.close()

print('\n\nDone')
