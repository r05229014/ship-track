import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from PIL import Image
from pathlib import Path
from pyhdf.SD import SD, SDC
from skimage.transform import resize
from mpl_toolkits.basemap import Basemap


def read_hdf(path):
    hdf = SD(path, SDC.READ)

    # read the data we choose
    scale = hdf.select('EV_500_Aggr1km_RefSB').attributes()['radiance_scales'][4]
    scale2 = hdf.select('EV_1KM_Emissive').attributes()['radiance_scales'][0]
    data = hdf.select('EV_500_Aggr1km_RefSB').get()[4] * scale
    data2 = hdf.select('EV_1KM_Emissive').get()[0] * scale2

    # read lon, lat
    lon, lat = hdf.select('Longitude').get(), hdf.select('Latitude').get()
    lon[lon < 0] = lon[lon < 0] + 360

    # resize
    lon = resize(lon, data.shape)
    lat = resize(lat, data.shape)
    return data, data2, lon, lat


def split_array(arr, size=(4, 3), axis=(0, 1)):
    arr_new = []
    arrs = np.split(arr, size[0], axis=axis[0])
    for arr in arrs:
        arr = np.array_split(arr, size[1], axis=axis[1])
        arr_new += (arr)
    return arr_new


def concat_arr2d(arr, size=(4, 3)):
    idx_arr = np.arange(0, 12).reshape(size)

    out = np.zeros((1, 1536))
    for i in range(size[0]):
        out = np.vstack((out, np.hstack(arr[idx_arr[i]])))
    return out


def ship_track_vis(var1, var2, predict, lon, lat, file_name):
    arrs = [var1, var2, predict]
    names = ['EV 500 Aggr1km RefSB', 'EV 1KM Emissive', 'Predict Ship Tracks']

    # process the longtide and latitude
    lon_min, lon_max = math.floor(lon.min()/10)*10, math.ceil(lon.max()/10)*10
    lat_min, lat_max = math.floor(lat.min()/10)*10, math.ceil(lat.max()/10)*10

    # plot
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))
    for i, ax in enumerate(axes.flat):
        map_ax = Basemap(ax=ax,
                         llcrnrlon=lon_min, urcrnrlon=lon_max,
                         llcrnrlat=lat_min, urcrnrlat=lat_max,
                         resolution='l')
        map_ax.drawcoastlines()
        map_ax.drawcoastlines(linewidth=1.3, color='limegreen')
        map_ax.drawparallels(np.arange(-100., 120., 10.), labels=[1, 0, 0, 0],
                             linewidth=1.5, dashes=(None, None),
                             color='lightgray', fontsize=18)
        map_ax.drawmeridians(np.arange(-180., 180., 10.), labels=[0, 0, 0, 1],
                             linewidth=1.5, dashes=(None, None),
                             color='lightgray', fontsize=18)
        # x, y = map_ax(lon, lat)
        map_ax.pcolormesh(lon, lat, arrs[i],
                          cmap=plt.cm.get_cmap('seismic'),
                          vmin=arrs[i].min(), vmax=arrs[i].max())
        ax.set_title(names[i], fontsize=25)
    plt.savefig(f'{file_name}.png')
