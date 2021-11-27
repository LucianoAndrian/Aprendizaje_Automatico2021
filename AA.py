# import
import os
import joblib
import matplotlib.pyplot as plt
plt.switch_backend('agg') #bug de pycharm...
os.environ['PROJ_LIB'] = '/home/auri/anaconda3/envs/py37/share/proj'
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
from itertools import groupby
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import statsmodels as sm
import xarray as xr

##--Functions--#########################################################################################################
def xrFieldTimeDetrend_sst(xrda, dim, deg=1):
    # detrend along a single dimension
    aux = xrda.polyfit(dim=dim, deg=deg, skipna=True)
    dt = xrda - xr.polyval(xrda[dim], aux.var_polyfit_coefficients[0])
    return dt

def MovingBasePeriodAnomaly(data, start='1920'):

    import xarray as xr
    # first five years
    start_num = int(start)

    initial = data.sel(time=slice(start + '-01-01', str(start_num + 5) + '-12-31')).groupby('time.month') - \
              data.sel(time=slice(str(start_num - 14) + '-01-01', str(start_num + 5 + 10) + '-12-31')).groupby(
                  'time.month').mean('time')


    start_num = start_num + 6
    result = initial

    while (start_num != 2016):

        aux = data.sel(time=slice(str(start_num) + '-01-01', str(start_num + 4) + '-12-31')).groupby('time.month') - \
              data.sel(time=slice(str(start_num - 15) + '-01-01', str(start_num + 4 + 10) + '-12-31')).groupby(
                  'time.month').mean('time')

        start_num = start_num + 5

        result = xr.concat([result, aux], dim='time')

    # 2016-2020 use base period 1991-2020
    aux = data.sel(time=slice(str(start_num) + '-01-01', str(start_num + 4) + '-12-31')).groupby('time.month') - \
          data.sel(time=slice('1991-01-01', '2020-12-31')).groupby('time.month').mean('time')

    result = xr.concat([result, aux], dim='time')

    return (result)

def ninioIndex(data, ninio=4,start='1950'):

    sst = data
    if ninio==12:
        #Niño 1+2 (0-10S, 90W-80W)
        # https://climatedataguide.ucar.edu/climate-data/ninio-sst-indices-ninio-12-3-34-4-oni-and-tni
        ninio_index = sst.sel(lat=slice(0, -10), lon=slice(270, 280), time=slice(start + '-01-01', '2020-12-31'))
    elif ninio== 4:
        #Niño 4 (5N-5S, 160E-150W)
        ninio_index = sst.sel(lat=slice(5, -5), lon=slice(160, 210), time=slice(start + '-01-01', '2020-12-31'))
    elif ninio==34:
        ninio_index = sst.sel(lat=slice(4.0, -4.0), lon=slice(190, 240), time=slice(start + '-01-01', '2020-12-31'))
    elif ninio == 3:
        ninio_index = sst.sel(lat=slice(4.0, -4.0), lon=slice(150, 270), time=slice(start + '-01-01', '2020-12-31'))

    # N34
    ninio_index = ninio_index['var'].mean(['lon', 'lat'], skipna=True)

    # compute monthly anomalies
    ninio_index = MovingBasePeriodAnomaly(ninio_index,start=start)

    # compute 5-month running mean
    ninio_index_filtered = np.convolve(ninio_index, np.ones((3,)) / 3, mode='same')  #
    ninio_index_f = xr.DataArray(ninio_index_filtered, coords=[ninio_index.time.values], dims=['time'])

    aux = abs(ninio_index_f) > 0.5
    results = []
    for k, g in groupby(enumerate(aux.values), key=lambda x: x[1]):
        if k:
            g = list(g)
            results.append([g[0][0], len(g)])

    n34 = []
    n34_df = pd.DataFrame(columns=['N34', 'Años', 'Mes'], dtype=float)
    for m in range(0, len(results)):
        # True values
        len_true = results[m][1]

        # True values for at least 5 consecutive seasons
        if len_true >= 5:
            a = results[m][0]
            n34.append([np.arange(a, a + results[m][1]), ninio_index_f[np.arange(a, a + results[m][1])].values])

            for l in range(0, len_true):
                if l < (len_true - 2):
                    main_month_num = results[m][0] + 1 + l
                    if main_month_num != 1210:
                        n34_df = n34_df.append({'N34': np.around(ninio_index_f[main_month_num].values, 2),
                                            'Años': np.around(ninio_index_f[main_month_num]['time.year'].values),
                                            'Mes': np.around(ninio_index_f[main_month_num]['time.month'].values)},
                                           ignore_index=True)

    return ninio_index_f, n34, n34_df

def KendallCorr_Lag(data, index, lag = 0):
    data = data.__mul__(1)
    index = index.__mul__(1)
    def Tau(x, y):
        from scipy import stats
        tau, pvalue = stats.kendalltau(x, y, nan_policy='propagate', method='asymptotic')
        return tau

    def TauSig(x, y):
        from scipy import stats
        tau, pvalue = stats.kendalltau(x, y, nan_policy='propagate', method='asymptotic')
        return pvalue

    def TauCorr(x, y, dim='time'):
        return xr.apply_ufunc(
            Tau, x, y,
            input_core_dims=[[dim], [dim]],
            vectorize=True,  # !Important!
            output_dtypes=[float])

    def TauCorrSig(x, y, dim='time'):
        return xr.apply_ufunc(
            TauSig, x, y,
            input_core_dims=[[dim], [dim]],
            vectorize=True,  # !Important!
            output_dtypes=[float])

    L = len(data.time)

    # para simplificar
    data['time'] = range(L)
    index['time'] = range(L)

    data = data.sel(time=slice(0,L-1-lag))
    index = index.sel(time=slice(0+lag,L-1))

    # para las funciones de Tau necesitan la misma
    # dim time
    data['time'] = range(len(data.time))
    index['time'] = range(len(data.time))

    tau = TauCorr(data, index);print('Corr Tau')
    sig = TauCorrSig(data, index);print('Corr Tau Sig')

    return xr.where(np.abs(sig)<0.01,tau, float('NaN'))

def SelectAreas0_1(data=None, step=2, threshold=0.1, n_areas=2):
    taus = data.__mul__(1) #que trucazo...
    taus['var'].values = np.abs(taus['var'].values)
    areas = pd.DataFrame(columns=['max_lon', 'min_lon', 'max_lat', 'min_lat', 'area'], dtype=float)

    for n in range(0,(n_areas)):
        aux = taus.where(taus == taus.max(), drop=True).squeeze()
        lon_c = aux.lon.values
        lat_c = aux.lat.values

        lon = lon_c
        while (taus.sel(lon=(lon), lat=lat_c)['var'].values > threshold) & \
                (np.isnan(taus.sel(lon=(lon), lat=lat_c)['var'].values)==False):
            lon = lon + step
            if lon == 360:
                lon = 0
        max_lon = lon


        lon = lon_c
        while (taus.sel(lon=(lon), lat=lat_c)['var'].values > threshold) & \
                (np.isnan(taus.sel(lon=(lon), lat=lat_c)['var'].values)==False):
            lon = lon - step
            if lon <0:
                lon = 358+lon
        min_lon = lon

        lat = lat_c
        while (taus.sel(lon=(lon_c), lat=(lat))['var'].values > threshold) & \
                (np.isnan(taus.sel(lon=(lon_c), lat=(lat))['var'].values)==False):
            lat = lat + step
        max_lat = lat

        lat = lat_c
        while (taus.sel(lon=(lon_c), lat=(lat))['var'].values > threshold) & \
                (np.isnan(taus.sel(lon=(lon_c), lat=(lat))['var'].values)==False):
            lat = lat - step
        min_lat = lat

        if max_lon < min_lon:
            areas = areas.append({'max_lon': max_lon, 'min_lon': 0,
                                 'max_lat': max_lat, 'min_lat': min_lat,
                                  'area': n + 1},
                                 ignore_index = True)

            taus.loc[dict(lon=slice(0, max_lon), lat=slice(max_lat, min_lat))] = float('NaN')

            areas = areas.append({'max_lon': 358, 'min_lon': min_lon,
                                 'max_lat': max_lat, 'min_lat': min_lat,
                                  'area': n + 1 + 1},
                                 ignore_index = True)

            taus.loc[dict(lon=slice(min_lon, 358), lat=slice(max_lat, min_lat))] = float('NaN')

        else:

            areas = areas.append({'max_lon': max_lon, 'min_lon': min_lon,
                                  'max_lat': max_lat, 'min_lat': min_lat,
                                  'area': n + 1},
                                 ignore_index=True)

            taus.loc[dict(lon=slice(min_lon, max_lon), lat=slice(max_lat, min_lat))] = float('NaN')



    return areas

def PlotAreas(aux, ssts, title):

    import cartopy.feature
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig = plt.figure(figsize=(7, 3.5), dpi=100)
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    crs_latlon = ccrs.PlateCarree()
    ax.set_extent([0, 359, -80, 80], crs=crs_latlon)

    im = ax.contourf(aux.lon, aux.lat, aux['var'], levels=np.linspace(-.5, .5, 11),
                     transform=crs_latlon, cmap='Spectral_r', extend='both')

    cb = plt.colorbar(im, fraction=0.042, pad=0.035, shrink=0.8)
    cb.ax.tick_params(labelsize=8)
    ax.add_feature(cartopy.feature.LAND, facecolor='#d9d9d9')
    ax.add_feature(cartopy.feature.COASTLINE)
    # ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax.gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-')
    ax.set_xticks(np.arange(30, 330, 60), crs=crs_latlon)
    ax.set_yticks(np.arange(-80, 80, 20), crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.tick_params(labelsize=7)

    for a in range(0,len(ssts)):
        ax.add_patch(
            patches.Rectangle(xy=(ssts.min_lon[a] - 180, ssts.max_lat[a]),
                              width=(ssts.max_lon[a] - ssts.min_lon[a]),
                              height=(ssts.min_lat[a] - ssts.max_lat[a]),
                              linewidth=1, color='black', fill=False)
        )

        centerx = np.mean([ssts.max_lon[a]-180, ssts.min_lon[a]-180])-5
        centery = np.mean([ssts.min_lat[a],ssts.max_lat[a]])-5
        plt.text(centerx, centery, 'Area: '+ str(a), size=6)
    plt.title(title)
    plt.savefig('./areas_features/'  + title + '.jpg')
    plt.close()

def Features(data, areas,omit_area=None, start_year = '1950', end_year='2020'):

    areas_num = np.arange(0,len(areas)).astype('float')
    if omit_area != None:
        print('Omit areas:' + str(omit_area))
        areas_num[omit_area] = np.nan
        areas_num = [x for x in areas_num if np.isnan(x) == False]

    for n in areas_num:
        aux = data.sel(lat=slice(areas.max_lat[n], areas.min_lat[n]), lon=slice(areas.min_lon[n], areas.max_lon[n])
                       , time=slice(start_year + '-01-01', end_year + '-12-31'))
        aux = aux['var'].mean(['lon', 'lat'], skipna=True)

        if n != areas_num[0]:
            aux2 = np.concatenate((aux2,np.array([aux.values])), axis=0)
        else:
            aux2 = np.array([aux.values])

    return aux2.T

def WaveFilter(serie, harmonic):

    import numpy as np

    sum = 0
    sam = 0
    N = np.size(serie)

    sum = 0
    sam = 0

    for j in range(N):
        sum = sum + serie[j] * np.sin(harmonic * 2 * np.pi * j / N)
        sam = sam + serie[j] * np.cos(harmonic * 2 * np.pi * j / N)

    A = 2*sum/N
    B = 2*sam/N

    xs = np.zeros(N)

    for j in range(N):
        xs[j] = A * np.sin(2 * np.pi * harmonic * j / N) + B * np.cos(2 * np.pi * harmonic * j / N)

    fil = serie - xs
    return(fil)

def DMIndex(iodw, iode, sst_anom_sd=True, xsd=0.5):

    import numpy as np
    from itertools import groupby
    import pandas as pd

    limitsize = len(iodw) - 2

    # dipole mode index
    dmi = iodw - iode

    # criteria
    western_sign = np.sign(iodw)
    eastern_sign = np.sign(iode)
    opposite_signs = western_sign != eastern_sign

    sd = np.std(dmi) * xsd
    print(str(sd))
    sdw = np.std(iodw.values) * xsd
    sde = np.std(iode.values) * xsd

    results = []
    for k, g in groupby(enumerate(opposite_signs.values), key=lambda x: x[1]):
        if k:
            g = list(g)
            results.append([g[0][0], len(g)])

    iods = pd.DataFrame(columns=['DMI', 'Años', 'Mes'], dtype=float)
    dmi_raw = []
    for m in range(0, len(results)):
        # True values
        len_true = results[m][1]

        # True values for at least 3 consecutive seasons
        if len_true >= 3:

            for l in range(0, len_true):

                if l < (len_true - 2):

                    main_month_num = results[m][0] + 1 + l
                    if main_month_num != limitsize:
                        main_month_name = dmi[main_month_num]['time.month'].values  # "name" 1 2 3 4 5

                        main_season = dmi[main_month_num]
                        b_season = dmi[main_month_num - 1]
                        a_season = dmi[main_month_num + 1]

                        # abs(dmi) > sd....(0.5*sd)
                        aux = (abs(main_season.values) > sd) & \
                              (abs(b_season) > sd) & \
                              (abs(a_season) > sd)

                        if sst_anom_sd:
                            if aux:
                                sstw_main = iodw[main_month_num]
                                sstw_b = iodw[main_month_num - 1]
                                sstw_a = iodw[main_month_num + 1]
                                #
                                aux2 = (abs(sstw_main) > sdw) & \
                                       (abs(sstw_b) > sdw) & \
                                       (abs(sstw_a) > sdw)
                                #
                                sste_main = iode[main_month_num]
                                sste_b = iode[main_month_num - 1]
                                sste_a = iode[main_month_num + 1]

                                aux3 = (abs(sste_main) > sde) & \
                                       (abs(sste_b) > sde) & \
                                       (abs(sste_a) > sde)

                                if aux3 & aux2:
                                    iods = iods.append({'DMI': np.around(dmi[main_month_num].values, 2),
                                                        'Años': np.around(dmi[main_month_num]['time.year'].values),
                                                        'Mes': np.around(dmi[main_month_num]['time.month'].values)},
                                                       ignore_index=True)

                                    a = results[m][0]
                                    dmi_raw.append([np.arange(a, a + results[m][1]),
                                                    dmi[np.arange(a, a + results[m][1])].values])


                        else:
                            if aux:
                                iods = iods.append({'DMI': np.around(dmi[main_month_num].values, 2),
                                                    'Años': np.around(dmi[main_month_num]['time.year'].values),
                                                    'Mes': np.around(dmi[main_month_num]['time.month'].values)},
                                                   ignore_index=True)

    return iods, dmi_raw

def DMI(per = 0, filter_bwa = True, filter_harmonic = True,
        filter_all_harmonic=True, harmonics = [],
        start_per=1920, end_per=2020):


    western_io = slice(50, 70) # definicion tradicional

    start_per = str(start_per)
    end_per = str(end_per)

    if per == 2:
        movinganomaly = True
        start_year = '1906'
        end_year = '2020'
        change_baseline = False
        start_year2 = '1920'
        end_year2 = '2020_30r5'
        print('30r5')
    else:
        movinganomaly = False
        start_year = start_per
        end_year = '2020'
        change_baseline = False
        start_year2 = '1920'
        end_year2 = end_per
        print('All')

    ##################################### DATA #####################################
    # ERSSTv5
    sst = xr.open_dataset("sst.mnmean.nc")
    dataname = 'ERSST'
    ##################################### Pre-processing #####################################
    iodw = sst.sel(lat=slice(10.0, -10.0), lon=western_io,
                       time=slice(start_year + '-01-01', end_year + '-12-31'))
    iodw = iodw.sst.mean(['lon', 'lat'], skipna=True)
    iodw2 = iodw
    if per == 2:
        iodw2 = iodw2[168:]
    # -----------------------------------------------------------------------------------#
    iode = sst.sel(lat=slice(0, -10.0), lon=slice(90, 110),
                   time=slice(start_year + '-01-01', end_year + '-12-31'))
    iode = iode.sst.mean(['lon', 'lat'], skipna=True)
    # -----------------------------------------------------------------------------------#
    bwa = sst.sel(lat=slice(20.0, -20.0), lon=slice(40, 110),
                  time=slice(start_year + '-01-01', end_year + '-12-31'))
    bwa = bwa.sst.mean(['lon', 'lat'], skipna=True)
    # ----------------------------------------------------------------------------------#

    if movinganomaly:
        iodw = MovingBasePeriodAnomaly(iodw)
        iode = MovingBasePeriodAnomaly(iode)
        bwa = MovingBasePeriodAnomaly(bwa)
    else:
        # change baseline
        if change_baseline:
            iodw = iodw.groupby('time.month') - \
                   iodw.sel(time=slice(start_year2 + '-01-01', end_year2 + '-12-31')).groupby('time.month').mean(
                       'time')

            iode = iode.groupby('time.month') - \
                   iode.sel(time=slice(start_year2 + '-01-01', end_year2 + '-12-31')).groupby('time.month').mean(
                       'time')

            bwa = bwa.groupby('time.month') - \
                  bwa.sel(time=slice(start_year2 + '-01-01', end_year2 + '-12-31')).groupby('time.month').mean(
                      'time')
            print('baseline: ' + str(start_year2) + ' - ' + str(end_year2))
        else:
            print('baseline: All period')
            iodw = iodw.groupby('time.month') - iodw.groupby('time.month').mean('time', skipna=True)
            iode = iode.groupby('time.month') - iode.groupby('time.month').mean('time', skipna=True)
            bwa = bwa.groupby('time.month') - bwa.groupby('time.month').mean('time', skipna=True)

    # ----------------------------------------------------------------------------------#
    # Detrend
    iodw_trend = np.polyfit(range(0, len(iodw)), iodw, deg=1)
    iodw = iodw - (iodw_trend[0] * range(0, len(iodw)) + iodw_trend[1])
    # ----------------------------------------------------------------------------------#
    iode_trend = np.polyfit(range(0, len(iode)), iode, deg=1)
    iode = iode - (iode_trend[0] * range(0, len(iode)) + iode_trend[1])
    # ----------------------------------------------------------------------------------#
    bwa_trend = np.polyfit(range(0, len(bwa)), bwa, deg=1)
    bwa = bwa - (bwa_trend[0] * range(0, len(bwa)) + bwa_trend[1])
    # ----------------------------------------------------------------------------------#

    # 3-Month running mean
    iodw_filtered = np.convolve(iodw, np.ones((3,)) / 3, mode='same')
    iode_filtered = np.convolve(iode, np.ones((3,)) / 3, mode='same')
    bwa_filtered = np.convolve(bwa, np.ones((3,)) / 3, mode='same')

    # Common preprocessing, for DMIs other than SY2003a
    iode_3rm = iode_filtered
    iodw_3rm = iodw_filtered

    #################################### follow SY2003a #######################################

    # power spectrum
    # aux = FFT2(iodw_3rm, maxVar=20, maxA=15).sort_values('Variance', ascending=False)
    # aux2 = FFT2(iode_3rm, maxVar=20, maxA=15).sort_values('Variance', ascending=False)

    # filtering harmonic
    if filter_harmonic:
        if filter_all_harmonic:
            for harmonic in range(15):
                iodw_filtered = WaveFilter(iodw_filtered, harmonic)
                iode_filtered = WaveFilter(iode_filtered, harmonic)
            else:
                for harmonic in harmonics:
                    iodw_filtered = WaveFilter(iodw_filtered, harmonic)
                    iode_filtered = WaveFilter(iode_filtered, harmonic)

    ## max corr. lag +3 in IODW
    ## max corr. lag +6 in IODE

    # ----------------------------------------------------------------------------------#
    # ENSO influence
    # pre processing same as before
    if filter_bwa:
        ninio3 = sst.sel(lat=slice(5.0, -5.0), lon=slice(210, 270),
                         time=slice(start_year + '-01-01', end_year + '-12-31'))
        ninio3 = ninio3.sst.mean(['lon', 'lat'], skipna=True)

        if movinganomaly:
            ninio3 = MovingBasePeriodAnomaly(ninio3)
        else:
            if change_baseline:
                ninio3 = ninio3.groupby('time.month') - \
                         ninio3.sel(time=slice(start_year2 + '-01-01', end_year2 + '-12-31')).groupby(
                             'time.month').mean(
                             'time')

            else:

                ninio3 = ninio3.groupby('time.month') - ninio3.groupby('time.month').mean('time', skipna=True)

            trend = np.polyfit(range(0, len(ninio3)), ninio3, deg=1)
            ninio3 = ninio3 - (trend[0] * range(0, len(ninio3)) +trend[1])

        # 3-month running mean
        ninio3_filtered = np.convolve(ninio3, np.ones((3,)) / 3, mode='same')

        # ----------------------------------------------------------------------------------#
        # removing BWA effect
        # lag de maxima corr coincide para las dos bases de datos.
        lag = 3
        x = pd.DataFrame({'bwa': bwa_filtered[lag:], 'ninio3': ninio3_filtered[:-lag]})
        result = sm.ols(formula='bwa~ninio3', data=x).fit()
        recta = result.params[1] * ninio3_filtered + result.params[0]
        iodw_f = iodw_filtered - recta

        lag = 6
        x = pd.DataFrame({'bwa': bwa_filtered[lag:], 'ninio3': ninio3_filtered[:-lag]})
        result = sm.ols(formula='bwa~ninio3', data=x).fit()
        recta = result.params[1] * ninio3_filtered + result.params[0]
        iode_f = iode_filtered - recta
        print('BWA filtrado')
    else:
        iodw_f = iodw_filtered
        iode_f = iode_filtered
        print('BWA no filtrado')
    # ----------------------------------------------------------------------------------#

    # END processing
    if movinganomaly:
        iodw_3rm = xr.DataArray(iodw_3rm, coords=[iodw.time.values], dims=['time'])
        iode_3rm = xr.DataArray(iode_3rm, coords=[iodw.time.values], dims=['time'])

        iodw_f = xr.DataArray(iodw_f, coords=[iodw.time.values], dims=['time'])
        iode_f = xr.DataArray(iode_f, coords=[iodw.time.values], dims=['time'])
        start_year = '1920'
    else:
        iodw_3rm = xr.DataArray(iodw_3rm, coords=[iodw2.time.values], dims=['time'])
        iode_3rm = xr.DataArray(iode_3rm, coords=[iodw2.time.values], dims=['time'])

        iodw_f = xr.DataArray(iodw_f, coords=[iodw2.time.values], dims=['time'])
        iode_f = xr.DataArray(iode_f, coords=[iodw2.time.values], dims=['time'])

    ####################################### compute DMI #######################################

    dmi_sy_full, dmi_raw = DMIndex(iodw_f, iode_f)

    return dmi_sy_full, dmi_raw, (iodw_f-iode_f)#, iodw_f - iode_f, iodw_f, iode_f


##--Datos--#############################################################################################################
path = 'ncfiles/'

#--SST--#
data = xr.open_dataset(path + 'sst.mnmean.nc')
data = data.sel(time=slice('1950-01-01', '2020-12-01'))
data = data.rename({'sst':'var'}) # para las funciones
data = xrFieldTimeDetrend_sst(data,'time')

# Indices según criterio CPC
ninio4 = ninioIndex(data, 4,'1950')[0]
ninio3 = ninioIndex(data, 3, '1950')[0]
#ninio12 = ninioIndex(data, 12, '1950')[0]
ninio34 = ninioIndex(data, 34, '1950')[0]
dmi = DMI(filter_bwa=False, per=0, start_per=1950)[2]

# Anomalia mensual
data = xr.open_dataset(path + 'sst.mnmean.nc')
data = data.sel(time=slice('1950-01-01', '2020-12-01'))
data = data.rename({'sst':'var'}) # para las funciones
data = xrFieldTimeDetrend_sst(data,'time')
data_prom = data.__mul__(1)
data = data.groupby('time.month') - data.groupby('time.month').mean()

#--HGT--#
data2 = xr.open_dataset(path +'ERA5_HGT.nc')
data2 = data2.sel(level=200)
data2 = data2.rename({'z':'var','longitude':'lon','latitude':'lat'}) # para las funciones
data2 = xrFieldTimeDetrend_sst(data2,'time')
data2_prom = data2.__mul__(1)
data2 = data2.groupby('time.month') - data2.groupby('time.month').mean()

#--viento_U--#
#850
data3 = xr.open_dataset(path + 'ERA5_U.nc')
data3 = data3.rename({'u':'var','longitude':'lon','latitude':'lat'})
data3a = data3.sel(level=850)
data3a = xrFieldTimeDetrend_sst(data3a,'time')
data3a_prom = data3a.__mul__(1)
data3a = data3a.groupby('time.month') - data3a.groupby('time.month').mean()
#200
data3b = data3.sel(level=200)
data3b = xrFieldTimeDetrend_sst(data3b,'time')
data3b_prom = data3b.__mul__(1)
data3b = data3b.groupby('time.month') - data3b.groupby('time.month').mean()
del data3

#--PSL--#
data4 = xr.open_dataset(path +'ERA5_SLP.nc')
data4 = data4.rename({'msl':'var','longitude':'lon','latitude':'lat'}) # para las funciones
data4 = xrFieldTimeDetrend_sst(data4,'time')
data4_prom = data4.__mul__(1)
data4 = data4.groupby('time.month') - data4.groupby('time.month').mean()

##--Development - Held Out--############################################################################################
data_ho = data.sel(time=slice('2011-01-01', '2020-12-01'))
data = data.sel(time=slice('1950-01-01', '2010-12-01'))

data2_ho = data2.sel(time=slice('2011-01-01', '2020-12-01'))
data2 = data2.sel(time=slice('1950-01-01', '2010-12-01'))

data3a_ho = data3a.sel(time=slice('2011-01-01', '2020-12-01'))
data3a = data3a.sel(time=slice('1950-01-01', '2010-12-01'))

data3b_ho = data3b.sel(time=slice('2011-01-01', '2020-12-01'))
data3b = data3b.sel(time=slice('1950-01-01', '2010-12-01'))

data4_ho = data4.sel(time=slice('2011-01-01', '2020-12-01'))
data4 = data4.sel(time=slice('1950-01-01', '2010-12-01'))

ninio4_ho = ninio4.sel(time=slice('2011-01-01', '2020-12-01'))
ninio4 = ninio4.sel(time=slice('1950-01-01', '2010-12-01'))

ninio3_ho = ninio3.sel(time=slice('2011-01-01', '2020-12-01'))
ninio3 = ninio3.sel(time=slice('1950-01-01', '2010-12-01'))

ninio34_ho = ninio34.sel(time=slice('2011-01-01', '2020-12-01'))
ninio34 = ninio34.sel(time=slice('1950-01-01', '2010-12-01'))

dmi_ho = dmi.sel(time=slice('2011-01-01', '2020-12-01'))
dmi = dmi.sel(time=slice('1950-01-01', '2010-12-01'))

aux = xr.DataArray(coords=[ninio4.time.values],dims=['time'])
aux['n4'] = ninio4
aux['n3'] = ninio3
aux['n34'] = ninio34
aux['ndmi'] = dmi
indices_name= ['n4','n3','n34','ndmi']

##--Seleccion de atributos--############################################################################################
#--Deteccion de regiones importantes para cada indice--#
def aux_areas(data, lags,i, name,n_areas=5, threshold=0.2):
    tau = KendallCorr_Lag(data, aux[indices_name[i]], lags)
    areas = SelectAreas0_1(data=tau, step=2, threshold=threshold, n_areas=n_areas)
    PlotAreas(tau, areas, name )
    areas.to_csv( 'aux_files/' + name +'.csv')

thr = [0.2,0.1] #menos exigencia para el lag de 6 meses
n_areas = [3,6] # más features para el lag de 6 meses
for i in range(0,4):
    n = 0
    if i == 3:
        thr[n] = 0.05; thr[n+1] = 0.05
    for lags in [3,6]:
        aux_areas(data, lags,i, str(indices_name[i]) + '_SST_' + str(lags),n_areas[n],thr[n])
        aux_areas(data2, lags, i, str(indices_name[i]) + '_HGT_' + str(lags),n_areas[n],thr[n])
        aux_areas(data3a, lags, i, str(indices_name[i]) + '_U850_' + str(lags),n_areas[n],thr[n])
        aux_areas(data3b, lags, i, str(indices_name[i]) + '_U200_' + str(lags),n_areas[n],thr[n])
        aux_areas(data4, lags, i, str(indices_name[i]) + '_PSL_' + str(lags),n_areas[n],thr[n])
        n = 1

##--Algoritmos--########################################################################################################

# Random Forest: tiende al overfitting si se permite "crecer" mucho los arboles
# SVR: overfitting si el epsilon es muy chico

# Sin tocar los demás hiperparametros:
# evaluacion en training y testing (held-out) de ambos algoritmos según max_depth para RF y epsilon para SVR

def PlotOverfittingAnalysis(model, lag, name='4', title=''):
    from sklearn.metrics import mean_squared_error
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    data_rs = pd.DataFrame(columns=['Training', 'Testing', 'Max_depth'], dtype=float)

    #----Instancias----#
    #traing
    areas_name = []
    for var in ['SST', 'HGT', 'U200', 'U850', 'PSL']:
        aux = pd.read_csv('aux_files/n' + name + '_' + var + '_' + str(lag) + '.csv')
        for m in aux['area']:
            areas_name.append(f'{var}' + f'{m}')

    aux = Features(data=data, areas=pd.read_csv('aux_files/n' + name + '_SST_' + str(lag) + '.csv'))
    aux2 = Features(data2, areas=pd.read_csv('aux_files/n' + name + '_HGT_' + str(lag) + '.csv'))
    aux3 = Features(data3a, areas=pd.read_csv('aux_files/n' + name + '_U200_' + str(lag) + '.csv'))
    aux4 = Features(data3b, areas=pd.read_csv('aux_files/n' + name + '_U850_' + str(lag) + '.csv'))
    aux5 = Features(data4, areas=pd.read_csv('aux_files/n' + name + '_PSL_' + str(lag) + '.csv'))
    features = np.concatenate((aux, aux2, aux3, aux4, aux5), axis=1)


    #held out testing
    aux = Features(data=data_ho, areas=pd.read_csv('aux_files/n' + name + '_SST_' + str(lag) + '.csv'))
    aux2 = Features(data2_ho, areas=pd.read_csv('aux_files/n' + name + '_HGT_' + str(lag) + '.csv'))
    aux3 = Features(data3a_ho, areas=pd.read_csv('aux_files/n' + name + '_U200_' + str(lag) + '.csv'))
    aux4 = Features(data3b_ho, areas=pd.read_csv('aux_files/n' + name + '_U850_' + str(lag) + '.csv'))
    aux5 = Features(data4_ho, areas=pd.read_csv('aux_files/n' + name + '_PSL_' + str(lag) + '.csv'))
    features_ho = np.concatenate((aux, aux2, aux3, aux4, aux5), axis=1)



    if name == '4':
        index = ninio4.values[lag:]
        index2 = ninio4_ho.values[lag:]
    elif name == '3':
        index = ninio3.values[lag:]
        index2 = ninio3_ho.values[lag:]
    elif name == '34':
        index = ninio34.values[lag:]
        index2 = ninio34_ho.values[lag:]
    elif name == 'dmi':
        index = dmi.values[lag:]
        index2 = dmi_ho.values[lag:]

    if model == 'RF':
        for n in range(2, 32, 2):
            RF = RandomForestRegressor(max_depth=n, n_jobs=-1,
                                       oob_score=True, random_state=42)

            RF.fit(features[:-lag], index)
            pred = RF.predict(features[:-lag])
            pred_ho = RF.predict(features_ho[:-lag])

            train = np.sqrt(mean_squared_error(index, pred))
            test = np.sqrt(mean_squared_error(index2, pred_ho))
            data_rs = data_rs.append({'Training': np.around(train, 4),
                                      'Testing': np.around(test, 4),
                                      'Max_depth': n}, ignore_index=True)

        fig, ax = plt.subplots()
        plt.plot(data_rs['Training'], label='Training')
        plt.plot(data_rs['Testing'], label='Testing')
        plt.legend()
        plt.ylim(-0.05, 0.8)
        plt.title('RMSE vs ' + title)
        plt.xticks(np.arange(0, len(data_rs), 1))
        plt.ylabel('ºC')
        anios = np.arange(2, 32, 2)
        ax = plt.gca()
        plt.axhline(y=0, color='gray')
        ax.set_xticklabels(anios)
        plt.grid(True)
        fig.set_size_inches(8, 4)
        plt.savefig('./salidas/RMSE' + name + '_' + str(lag) + 'RF_test')
        plt.close()

    else:
        for n in [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1]:

            model = make_pipeline(StandardScaler(), SVR(epsilon=n))
            SVR_reg = TransformedTargetRegressor(regressor=model, transformer=StandardScaler())
            SVR_reg.fit(features[:-lag], index)

            pred = SVR_reg.predict(features[:-lag])
            pred_ho = SVR_reg.predict(features_ho[:-lag])

            train = np.sqrt(mean_squared_error(index, pred))
            test = np.sqrt(mean_squared_error(index2, pred_ho))
            data_rs = data_rs.append({'Training': np.around(train, 4),
                                      'Testing': np.around(test, 4),
                                      'Max_depth': n}, ignore_index=True)

        fig, ax = plt.subplots()
        plt.plot(data_rs['Training'], label='Training')
        plt.plot(data_rs['Testing'], label='Testing')
        plt.legend()
        plt.ylim(-0.05, 0.8)
        plt.title('RMSE vs ' + title)
        plt.xticks(np.arange(0, len(data_rs), 1))
        plt.ylabel('ºC')
        anios = [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
        ax = plt.gca()
        plt.axhline(y=0, color='gray')
        ax.set_xticklabels(anios)
        plt.grid(True)
        fig.set_size_inches(8, 4)
        plt.savefig('./salidas/RMSE' + name + '_' + str(lag) + 'SVR_test')
        plt.close()

PlotOverfittingAnalysis('RF',3,'4','max_depth - Niño4')
PlotOverfittingAnalysis('RF',6,'4','max_depth - Niño4')
PlotOverfittingAnalysis('RF',3,'3','max_depth - Niño3')
PlotOverfittingAnalysis('RF',6,'3','max_depth - Niño3')
PlotOverfittingAnalysis('RF',3,'34','max_depth - Niño3')
PlotOverfittingAnalysis('RF',6,'34','max_depth - Niño3')
PlotOverfittingAnalysis('RF',3,'dmi','max_depth - DMI')
PlotOverfittingAnalysis('RF',6,'dmi','max_depth - DMI')


PlotOverfittingAnalysis('SVR',3,'4','epsilon - Niño4')
PlotOverfittingAnalysis('SVR',6,'4','epsilon - Niño4')
PlotOverfittingAnalysis('SVR',3,'3','epsilon - Niño3')
PlotOverfittingAnalysis('SVR',6,'3','epsilon - Niño3')
PlotOverfittingAnalysis('SVR',3,'34','epsilon - Niño34')
PlotOverfittingAnalysis('SVR',6,'34','epsilon - Niño34')
PlotOverfittingAnalysis('SVR',3,'dmi','max_depth - DMI')
PlotOverfittingAnalysis('SVR',6,'dmi','max_depth - DMI')


# Usando max_depth y epsilon donde se minimiza la diferencia entre training y testing evitando el overfitting

########################################################################################################################
##--Tunning de hiperparametros--########################################################################################
########################################################################################################################
##--Random Forest--#####################################################################################################
def RF_RandSearch_Training(name,lag,max_depth=4, title='Niño'):

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import RandomizedSearchCV
    import joblib

    data_rs = pd.DataFrame(columns=['Training', 'Testing', 'Max_depth'], dtype=float)

    #-- Instancias training --#
    areas_name = []
    for var in ['SST', 'HGT', 'U200', 'U850', 'PSL']:
        aux = pd.read_csv('aux_files/n' + name + '_' + var + '_' + str(lag) + '.csv')
        for m in aux['area']:
            areas_name.append(f'{var}' + f'{m}')

    aux = Features(data=data, areas=pd.read_csv('aux_files/n' + name + '_SST_' + str(lag) + '.csv'))
    aux2 = Features(data2, areas=pd.read_csv('aux_files/n' + name + '_HGT_' + str(lag) + '.csv'))
    aux3 = Features(data3a, areas=pd.read_csv('aux_files/n' + name + '_U200_' + str(lag) + '.csv'))
    aux4 = Features(data3b, areas=pd.read_csv('aux_files/n' + name + '_U850_' + str(lag) + '.csv'))
    aux5 = Features(data4, areas=pd.read_csv('aux_files/n' + name + '_PSL_' + str(lag) + '.csv'))
    features = np.concatenate((aux, aux2, aux3, aux4, aux5), axis=1)

    # Random Forest
    RF_reg = RandomForestRegressor(n_jobs=-1, oob_score=True,
                                   max_depth=max_depth, random_state=42)
    # Random Search
    param_grid = {
        'n_estimators': [350, 400, 450, 500, 550, 600, 650, 750],
        'min_samples_leaf': [2, 4, 8, 10, 16, 24, 32, 42],
        'min_samples_split': [4, 6, 8, 10, 12, 24, 32, 42],
        'max_leaf_nodes': [30, 35, 40, 45, 50, 55, 60, 65, 70, 75],
        'max_features': ['auto', 'sqrt', 'log2']
    }
    # cv = 5 default k-fold
    random_search = RandomizedSearchCV(estimator=RF_reg,
                                       param_distributions=param_grid,
                                       n_iter=200, verbose=1, scoring='neg_root_mean_squared_error',
                                       random_state=42, n_jobs=-1, return_train_score=True)
    # ...
    if name == '4':
        gd = random_search.fit(features[:-lag], ninio4.values[lag:])
        index = ninio4.values
        index2 = ninio4_ho.values
    elif name == '3':
        gd = random_search.fit(features[:-lag], ninio3.values[lag:])
        index = ninio3.values
        index2 = ninio3_ho.values
    elif name == '34':
        gd = random_search.fit(features[:-lag], ninio34.values[lag:])
        index = ninio34.values
        index2 = ninio34_ho.values
    elif name == 'dmi':
        gd = random_search.fit(features[:-lag], dmi.values[lag:])
        index = dmi.values
        index2 = dmi_ho.values


    cv_results = pd.DataFrame(gd.cv_results_)
    data_rs = data_rs.append({'Training': np.around(cv_results['mean_train_score'].max(), 4),
                              'Testing': np.around(cv_results['mean_test_score'].max(), 4)},
                             ignore_index=True)
    print(data_rs)
    data_rs.to_csv('aux_files/' + 'Tunning_RF_' + name + '_' + str(lag) + '.csv')

    # Mejor Estimador a patir de Random Search
    RF_reg_best = gd.best_estimator_

    # save
    joblib.dump(RF_reg_best, './modelos/RF_reg_best_' + name + '_' + str(lag) + '.joblib')

    # Skill en Training
    pred = RF_reg_best.predict(features[:-lag])

    fig, ax = plt.subplots()
    plt.plot(index[lag:], label=title + name)
    from sklearn.metrics import explained_variance_score
    ev = explained_variance_score(index[lag:], pred)
    from sklearn.metrics import mean_squared_error
    mse = np.sqrt(mean_squared_error(index[lag:], pred))

    plt.plot(pred, label='RF_predict')
    plt.legend()
    plt.ylim(-1.5, 1.5)
    plt.title(title + name + ' '+ 'Lag-'  + str(lag) + '\n RMSE ' +
              str(np.around(mse, 3)) + ' - ' + 'EV ' + str(np.around(ev, 3)))
    plt.xticks(np.arange(0, len(index) + 12, 60), rotation=45)
    plt.ylabel('ºC')
    anios = np.arange(1950, 2011, 5)
    ax = plt.gca()
    ax.set_xticklabels(anios)
    plt.grid(True)
    plt.savefig('./salidas/' + title +  name + '_' + str(lag) + 'RF_training')
    fig.set_size_inches(3, 7)
    plt.close()

    # Testing Held-Out
    #-- Instancias testing --#
    areas_name = []
    for var in ['SST', 'HGT', 'U200', 'U850', 'PSL']:
        aux = pd.read_csv('aux_files/n' + name + '_' + var + '_' + str(lag) + '.csv')
        for m in aux['area']:
            areas_name.append(f'{var}' + f'{m}')

    aux = Features(data=data_ho, areas=pd.read_csv('aux_files/n' + name + '_SST_' + str(lag) + '.csv'))
    aux2 = Features(data2_ho, areas=pd.read_csv('aux_files/n' + name + '_HGT_' + str(lag) + '.csv'))
    aux3 = Features(data3a_ho, areas=pd.read_csv('aux_files/n' + name + '_U200_' + str(lag) + '.csv'))
    aux4 = Features(data3b_ho, areas=pd.read_csv('aux_files/n' + name + '_U850_' + str(lag) + '.csv'))
    aux5 = Features(data4_ho, areas=pd.read_csv('aux_files/n' + name + '_PSL_' + str(lag) + '.csv'))
    features_ho = np.concatenate((aux, aux2, aux3, aux4, aux5), axis=1)

    # Skill en testing
    index_pred = RF_reg_best.predict(features_ho[:-lag])

    fig, ax = plt.subplots()
    plt.plot(index2[lag:], label=title + name)
    from sklearn.metrics import explained_variance_score
    ev = explained_variance_score(index2[lag:], index_pred)
    from sklearn.metrics import mean_squared_error
    mse = np.sqrt(mean_squared_error(index2[lag:], index_pred))

    plt.plot(index_pred, label='RF_predict')
    plt.legend()
    plt.ylim(-1.5, 1.5)
    plt.title(title + name + ' ' + 'Lag-' + str(lag) + '\n RMSE ' +
              str(np.around(mse, 3)) + ' - ' + 'EV ' + str(np.around(ev, 3)))
    plt.xticks(np.arange(0, len(index_pred) + 12, 12), rotation=45)
    plt.ylabel('ºC')
    anios = np.arange(2011, 2022, 1)
    ax = plt.gca()
    plt.axhline(y=0, color='gray')
    ax.set_xticklabels(anios)
    plt.grid(True)
    plt.savefig('./salidas/' + title + name + '_' + str(lag) + 'RF_test')
    fig.set_size_inches(3, 7)
    plt.close()

RF_RandSearch_Training('4',3,6)
RF_RandSearch_Training('4',6,4)
RF_RandSearch_Training('3',3,6)
RF_RandSearch_Training('3',6,4)
RF_RandSearch_Training('34',3,5)
RF_RandSearch_Training('34',6,4)
RF_RandSearch_Training('dmi',3,4,title='DMI')
RF_RandSearch_Training('dmi',6,6,title='DMI')

##--Support Vector Machine Regressor--##################################################################################
def SVR_RandSearch_Training(name,lag,epsilon=0.1,title='Ninio'):

    from sklearn.svm import SVR
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import RandomizedSearchCV
    import joblib

    data_rs = pd.DataFrame(columns=['Training', 'Testing', 'Max_depth'], dtype=float)

    #-- Instancias training --#
    areas_name = []
    for var in ['SST', 'HGT', 'U200', 'U850', 'PSL']:
        aux = pd.read_csv('aux_files/n' + name + '_' + var + '_' + str(lag) + '.csv')
        for m in aux['area']:
            areas_name.append(f'{var}' + f'{m}')

    aux = Features(data=data, areas=pd.read_csv('aux_files/n' + name + '_SST_' + str(lag) + '.csv'))
    aux2 = Features(data2, areas=pd.read_csv('aux_files/n' + name + '_HGT_' + str(lag) + '.csv'))
    aux3 = Features(data3a, areas=pd.read_csv('aux_files/n' + name + '_U200_' + str(lag) + '.csv'))
    aux4 = Features(data3b, areas=pd.read_csv('aux_files/n' + name + '_U850_' + str(lag) + '.csv'))
    aux5 = Features(data4, areas=pd.read_csv('aux_files/n' + name + '_PSL_' + str(lag) + '.csv'))
    features = np.concatenate((aux, aux2, aux3, aux4, aux5), axis=1)

    # Support Vector Machine - Regressor
    SVR_reg = SVR(epsilon=epsilon)
    # Random Search
    param_grid = {
        'C': [1, 5, 10, 15, 20, 25,30,35],
        'degree': [3, 5, 10, 15, 20, 25, 30, 35, 40],
        "gamma": [0.0000001, 0.001, 0.005, 0.009, 0.01, 0.05],
    }

    # cv = 5 default k-fold
    random_search = RandomizedSearchCV(estimator=SVR_reg,
                                       param_distributions=param_grid,
                                       n_iter=200, verbose=1, scoring='neg_root_mean_squared_error',
                                       random_state=42, n_jobs=-1, return_train_score=True)

    random_search = make_pipeline(StandardScaler(), random_search)

    if name == '4':
        gd = random_search.fit(features[:-lag], ninio4.values[lag:])
        index = ninio4.values
        index2 = ninio4_ho.values
    elif name == '3':
        gd = random_search.fit(features[:-lag], ninio3.values[lag:])
        index = ninio3.values
        index2 = ninio3_ho.values
    elif name == '34':
        gd = random_search.fit(features[:-lag], ninio3.values[lag:])
        index = ninio34.values
        index2 = ninio34_ho.values
    elif name == 'dmi':
        gd = random_search.fit(features[:-lag], dmi.values[lag:])
        index = dmi.values
        index2 = dmi_ho.values


    cv_results = pd.DataFrame(gd[-1].cv_results_)
    data_rs = data_rs.append({'Training': np.around(cv_results['mean_train_score'].max(), 4),
                              'Testing': np.around(cv_results['mean_test_score'].max(), 4)},
                             ignore_index=True)
    print(data_rs)
    data_rs.to_csv('./aux_files/' + 'Tunning_SVR_' + name + '_' + str(lag) + '.csv')

    # Mejor Estimador a patir de Random Search
    model = make_pipeline(StandardScaler(), gd[-1].best_estimator_)
    SVR_reg_best = TransformedTargetRegressor(regressor=model, transformer=StandardScaler())
    SVR_reg_best = SVR_reg_best.fit(features[:-lag], index[lag:])

    # save
    joblib.dump(SVR_reg_best, './modelos/SVR_reg_best_' + name + '_' + str(lag) + '.joblib')

    pred = SVR_reg_best.predict(features[:-lag])

    fig, ax = plt.subplots()
    plt.plot(index[lag:], label=title + name)
    from sklearn.metrics import explained_variance_score
    ev = explained_variance_score(index[lag:], pred)
    from sklearn.metrics import mean_squared_error
    mse = np.sqrt(mean_squared_error(index[lag:], pred))

    plt.plot(pred, label='SVR_predict')
    plt.legend()
    plt.ylim(-1.5, 1.5)
    plt.title(title + name + ' '+ 'Lag-'  + str(lag) + '\n RMSE ' +
              str(np.around(mse, 3)) + ' - ' + 'EV ' + str(np.around(ev, 3)))
    plt.xticks(np.arange(0, len(index) + 12, 60), rotation=45)
    plt.ylabel('ºC')
    anios = np.arange(1950, 2011, 5)
    ax = plt.gca()
    ax.set_xticklabels(anios)
    plt.grid(True)
    plt.savefig('./salidas/' + title + name + '_' + str(lag) + 'SVR_training')
    fig.set_size_inches(3, 7)
    plt.close()

    import joblib
    # save
    joblib.dump(SVR_reg_best, './SVR_reg_best_' + name + '.joblib')

    areas_name = []
    for var in ['SST', 'HGT', 'U200', 'U850', 'PSL']:
        aux = pd.read_csv('aux_files/n' + name + '_' + var + '_' + str(lag) + '.csv')
        for m in aux['area']:
            areas_name.append(f'{var}' + f'{m}')

    aux = Features(data=data_ho, areas=pd.read_csv('aux_files/n' + name + '_SST_' + str(lag) + '.csv'))
    aux2 = Features(data2_ho, areas=pd.read_csv('aux_files/n' + name + '_HGT_' + str(lag) + '.csv'))
    aux3 = Features(data3a_ho, areas=pd.read_csv('aux_files/n' + name + '_U200_' + str(lag) + '.csv'))
    aux4 = Features(data3b_ho, areas=pd.read_csv('aux_files/n' + name + '_U850_' + str(lag) + '.csv'))
    aux5 = Features(data4_ho, areas=pd.read_csv('aux_files/n' + name + '_PSL_' + str(lag) + '.csv'))
    features_ho = np.concatenate((aux, aux2, aux3, aux4, aux5), axis=1)

    # Skill en testing
    index_pred = SVR_reg_best.predict(features_ho)

    fig, ax = plt.subplots()
    plt.plot(index2, label=title + name)
    from sklearn.metrics import explained_variance_score
    ev = explained_variance_score(index2, index_pred)
    from sklearn.metrics import mean_squared_error
    mse = np.sqrt(mean_squared_error(index2, index_pred))

    plt.plot(index_pred, label='SVR_predict')
    plt.legend()
    plt.ylim(-1.5, 1.5)
    plt.title(title + name + ' '  + 'Lag-' + str(lag) + '\n RMSE ' +
              str(np.around(mse, 3)) + ' - ' + 'EV ' + str(np.around(ev, 3)))
    plt.xticks(np.arange(0, len(index_pred) + 12, 12), rotation=45)
    plt.ylabel('ºC')
    anios = np.arange(2011, 2022, 1)
    ax = plt.gca()
    plt.axhline(y=0, color='gray')
    ax.set_xticklabels(anios)
    plt.grid(True)
    plt.savefig('./salidas/' + title + name + '_' + str(lag) + 'SVR_test')
    fig.set_size_inches(3, 7)
    plt.close()

SVR_RandSearch_Training('4',3,epsilon=0.7)
SVR_RandSearch_Training('4',6,epsilon=0.7)
SVR_RandSearch_Training('3',3,epsilon=0.5)
SVR_RandSearch_Training('3',6,epsilon=0.7)
SVR_RandSearch_Training('34',3,epsilon=0.5)
SVR_RandSearch_Training('34',6,epsilon=0.9)
SVR_RandSearch_Training('dmi',3,epsilon=0.5, title='DMI')
SVR_RandSearch_Training('dmi',6,epsilon=0.5, title='DMI')

########################################################################################################################