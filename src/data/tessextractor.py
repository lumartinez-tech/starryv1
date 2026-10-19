import numpy as np
import scipy
from astroquery.mast import Catalogs
from astroquery.mast import Tesscut
Tesscut.TIMEOUT = 60
from astroquery.mast import TesscutClass
from astroquery.skyview import SkyView
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import ascii
import time
from numpy import median
from PyAstronomy.pyasl import foldAt
#from photutils import CircularAperture
from photutils.aperture import CircularAperture
from photutils.aperture import CircularAnnulus
from photutils.aperture import aperture_photometry
from astropy.stats import sigma_clipped_stats
from scipy.stats import mode
from scipy.interpolate import interp1d
from astropy.stats import median_absolute_deviation
from scipy.stats import linregress
import pandas as pd
from astropy.timeseries import LombScargle
import astropy.units as u
import concurrent.futures
import warnings
#warnings.warn("InputWarning: Coordinate string is being interpreted as an ICRS coordinate provided in degrees. [astroquery.utils.commons]")
warnings.filterwarnings("ignore")
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

"""#GAMMAPY Robust Periodogram


"""

# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import scipy.optimize

__all__ = ["robust_periodogram", "plot_periodogram"]

def robust_periodogram(time, flux, flux_err=None, periods=None, loss="linear", scale=1):
    """
    Compute a light curve's period.

    A single harmonic model is fitted to the light curve.
    The periodogram returns the power for each period.
    The maximum power indicates the period of the light curve, assuming an underlying periodic process.

    The fitting can be done by ordinary least square regression (Lomb-Scargle periodogram) or robust regression.
    For robust regression, the scipy object `~scipy.optimize.least_squares` is called.
    For an introduction to robust regression techniques and loss functions, see [1]_ and [2]_.

    The significance of a periodogram peak can be evaluated in terms of a false alarm probability.
    It can be computed with the `~false_alarm_probability`-method of `~astropy`, assuming Gaussian white noise light curves.
    For an introduction to the false alarm probability of periodogram peaks, see :ref:`stats-lombscargle`.

    The periodogram is biased by measurement errors, high order modes and sampling of the light curve.
    To evaluate the impact of the sampling, compute the spectral window function with the `astropy.timeseries.LombScargle` class.

    The function returns a dictionary with the following content:

    - ``periods`` (`numpy.ndarray`) -- Period grid in units of ``t``
    - ``power`` (`numpy.ndarray`) -- Periodogram peaks at periods of ``pgrid``
    - ``best_period`` (float) -- Period of the highest periodogram peak

    Parameters
    ----------
    time : `numpy.ndarray`
        Time array of the light curve
    flux : `numpy.ndarray`
        Flux array of the light curve
    flux_err : `numpy.ndarray`
        Flux error array of the light curve. Default is 1.
    periods : `numpy.ndarray`
        Period grid on which the periodogram is performed.
        If not given, a linear grid will be computed limited by the length of the light curve and the Nyquist frequency.
    loss : {'linear', 'soft_l1', 'huber', 'cauchy', 'arctan'}
        Loss function for the robust regression.
        Default is 'linear', resulting in the Lomb-Scargle periodogram.
    scale : float (optional, default=1)
        Loss scale parameter to define margin between inlier and outlier residuals.
        If not given, will be set to 1.

    Returns
    -------
    results : `dict`
        Results dictionary (see description above).

    References
    ----------
    .. [1] Nikolay Mayorov (2015), "Robust nonlinear regression in scipy",
       see `here <http://scipy-cookbook.readthedocs.io/items/robust_regression.html>`__
    .. [2] Thieler et at. (2016), "RobPer: An R Package to Calculate Periodograms for Light Curves Based on Robust Regression",
       see `here <https://www.jstatsoft.org/article/view/v069i09>`__
    """
    if flux_err is None:
        flux_err = np.ones_like(flux)

    # set up period grid
    if periods is None:
        periods = _period_grid(time)

    # compute periodogram
    psd_data = _robust_regression(time, flux, flux_err, periods, loss, scale)

    # find period of highest periodogram peak
    best_period = periods[np.argmax(psd_data)]

    return {"periods": periods, "power": psd_data, "best_period": best_period}



def _period_grid(time):
    """
    Generates the period grid for the periodogram
    """
    number_obs = len(time)
    length_lc = np.max(time) - np.min(time)

    dt = 2 * length_lc / number_obs
    max_period = np.rint(length_lc / dt) * dt
    min_period = dt

    periods = np.arange(min_period, max_period + dt, dt)

    return periods


def _model(beta0, x, period, t, y, dy):
    """
    Computes the residuals of the periodic model
    """
    x[:, 1] = np.cos(2 * np.pi * t / period)
    x[:, 2] = np.sin(2 * np.pi * t / period)

    return (y - np.dot(x, beta0.T)) / dy


def _noise(mu, t, y, dy):
    """
    Residuals of the noise-only model.
    """
    return (mu * np.ones(len(t)) - y) / dy


def _robust_regression(time, flux, flux_err, periods, loss, scale):
    """
    Periodogram peaks for a given loss function and scale.
    """
    beta0 = np.array([0, 1, 0])
    mu = np.median(flux)
    x = np.ones([len(time), len(beta0)])
    chi_model = np.empty([len(periods)])
    chi_noise = np.empty([len(periods)])

    for i in range(len(periods)):
        chi_model[i] = scipy.optimize.least_squares(
            _model,
            beta0,
            loss=loss,
            f_scale=scale,
            args=(x, periods[i], time, flux, flux_err),
        ).cost
        chi_noise[i] = scipy.optimize.least_squares(
            _noise, mu, loss=loss, f_scale=scale, args=(time, flux, flux_err)
        ).cost
    power = 1 - chi_model / chi_noise

    return power


def plot_periodogram(
    time, flux, periods, power, flux_err=None, best_period=None, fap=None
):
    """
    Plot a light curve and its periodogram.

    The highest period of the periodogram and its false alarm probability (FAP) is added to the plot, if given.

    Parameters
    ----------
    time : `numpy.ndarray`
        Time array of the light curve
    flux : `numpy.ndarray`
        Flux array of the light curve
    periods : `numpy.ndarray`
        Periods for the periodogram
    power : `numpy.ndarray`
        Periodogram peaks of the data
    flux_err : `numpy.ndarray` (optional, default=None)
        Flux error array of the light curve.
        Is set to 0 if not given.
    best_period : float (optional, default=None)
        Period of the highest periodogram peak
    fap : float (optional, default=None)
        False alarm probability of ``best_period`` under a certain significance criterion.

    Returns
    -------
    fig : `~matplotlib.figure.Figure`
        Matplotlib figure
    """
    import matplotlib.pyplot as plt

    if flux_err is None:
        flux_err = np.zeros_like(flux)

    # set up the figure & axes for plotting
    fig = plt.figure(figsize=(16, 9))
    grid_spec = plt.GridSpec(2, 1)

    # plot the light curve
    ax = fig.add_subplot(grid_spec[0, :])
    ax.errorbar(
        time, flux, flux_err, fmt="ok", label="light curve", elinewidth=1.5, capsize=0
    )
    ax.set_xlabel("time")
    ax.set_ylabel("flux")
    ax.legend()

    # plot the periodogram
    ax = fig.add_subplot(grid_spec[1, :])
    ax.plot(periods, power, c="k", label="periodogram")
    # mark the best period and label with significance
    if best_period is not None:
        if fap is None:
            raise ValueError(
                "Must give a false alarm probability if you give a best_period"
            )

        # set precision for period format
        pre = int(abs(np.floor(np.log10(np.max(np.diff(periods))))))
        label = "Detected period p = {:.{}f} with {:.2E} FAP".format(
            best_period, pre, fap
        )
        ymax = power[periods == best_period]
        ax.axvline(best_period, ymin=0, ymax=ymax, label=label, c="r")

    ax.set_xlabel("period")
    ax.set_ylabel("power")
    ax.set_xlim(0, np.max(periods))
    ax.legend()

    return fig

"""#Pipeline"""

output_file = os.path.join(BASE_DIR, 'OrionL_sample1_TESSextractor_output.txt')
f = open(output_file, "w+")

f.write("#TIC Tmag ra dec per fm Delta total_flux_ratio sector\n")

targets_file = os.path.join(BASE_DIR, 'Orion_input_tess')
star = np.genfromtxt(targets_file, comments='#', dtype='S')
tic=star[:,0].astype(int)
Tmag=star[:,1].astype(float)
sector=star[:,2].astype(int)


def plot_cutout(image):
    """
   # Plot image and add grid lines.
   # """
    plt.imshow(image, origin = 'lower', cmap = plt.cm.YlGnBu_r,
           vmax = np.percentile(image, 90),
           vmin = np.percentile(image, 5))

def crowdinglevel(ra,dec,ap1,ap2,ap3):

	AP=ap1*0.00583333
	AP2=ap2*0.00583333
	AP3=ap3*0.00583333
	APsky=5*0.00583333

	catalogData = Catalogs.query_region("%.6f %.6f" %(ra, dec), radius=AP, catalog = "gaia", version=2)
	gmag = catalogData['phot_g_mean_mag']
	flux=10**(-0.4*gmag)
	catalogData2=Catalogs.query_region("%.6f %.6f" %(ra, dec), radius=AP2, catalog="gaia", version=2)
	catalogData3=Catalogs.query_region("%.6f %.6f" %(ra, dec), radius=AP3, catalog="gaia", version=2)
	df2 = pd.DataFrame(np.array(catalogData2), columns=["ra","dec","phot_g_mean_mag","distance"])
	df3 = pd.DataFrame(np.array(catalogData3), columns=["ra","dec","phot_g_mean_mag","distance"])
	df=pd.merge(df3, df2, on=['ra','dec'], how='outer', indicator=True)
	df=df[df['_merge'] == 'left_only']
	gmag_sky=df['phot_g_mean_mag_x']
	flux_sky=10**(-0.4*gmag_sky)

	catalogDatasky=Catalogs.query_region("%.6f %.6f" %(ra, dec), radius=APsky, catalog="gaia", version=2)
	gmagtotal= catalogDatasky['phot_g_mean_mag']
	fluxtotal=10**(-0.4*gmagtotal)
	return sum(flux[1:])/flux[0], sum(flux[1:]), len(gmag[1:]), sum(flux_sky[:])/flux[0], sum(flux_sky[:]), len(gmag_sky[:]), sum(fluxtotal[1:])/fluxtotal[0], len(gmagtotal[1:])

import signal
class TimeoutException(Exception):
    pass
def handler(signum, frame):
    raise TimeoutException()


def Tess_Period(tic, Tmag, sector):
    starName = "TIC " + str(tic)
    catalogData = Catalogs.query_object(starName, catalog = "TIC")
    ra = catalogData[0]['ra']
    dec = catalogData[0]['dec']

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(20) #seconds

    try:
        sky= SkyView.get_images(position='%.5f %.5f' % (ra, dec), survey=['DSS2 IR'], pixels=210, show_progress=True) #DSS2 1arcsec por pix, DSS 1.7arcsec por pix
        sv=sky[0]
        image_data = sv[0].data
        skyview_available = True
    except TimeoutException:
        print("SkyView failed")
        skyview_available = False
    except Exception as e:
        print("SkyView failed {e}")
        skyview_available = False
    finally:
        signal.alarm(0) #alarm is desactivated

    coord = SkyCoord(ra, dec, unit = "deg")

    try:
        Tesscut.TIMEOUT = 60
        hdulist = TesscutClass().get_cutouts(objectname=starName, size=10, sector=sector)
    except Exception as e:
        print(f"❌ Tesscut failed for TIC {tic} in sector {sector}: {e}")
        return None

#    hdulist[0].info()
    hdulist[0][0].header['SECTOR']
#    hdulist[1].info()

    # Define a function to simplify the plotting command that we do repeatedly.

    hdu = hdulist[0]
    firstImage = hdu[1].data['FLUX'][0]

    #fig = plt.figure(figsize=(7, 7))
    #plt.savefig("TIC_%i_Tmag_%.2f.png" % (tic, Tmag))

    #hdu2 = hdulist[1]

    #firstImage = hdu2[1].data['FLUX'][0]

    #wcs = WCS(hdu2[2].header)

    #fig = plt.figure(figsize = (8, 8))
    #fig.add_subplot(111, projection = wcs)
    #plot_cutout(firstImage)

    #plt.xlabel('RA', fontsize = 12)
    #plt.ylabel('Dec', fontsize = 12)

    #starloc = wcs.all_world2pix([[ra,dec]],0)  #Second is origin
    #plt.scatter(starloc[0,0], starloc[0,1],s = 45,color = 'red')

    wcs = WCS(hdu[2].header)

    #fig = plt.figure(figsize = (8, 8))
    #fig.add_subplot(111, projection = wcs)
    #plot_cutout(firstImage)

    #plt.xlabel('RA', fontsize = 12)
    #plt.ylabel('Dec', fontsize = 12)

    starloc = wcs.all_world2pix([[ra,dec]],0)  #Second is origin
    if (Tmag>13):

        aperturemin = CircularAperture(starloc, r=1)
        aperture = CircularAperture(starloc, r=1.5)
        pixels1=1.5
        pixels2=2.5
        pixels3=3.5
        annulus_aperture = CircularAnnulus(starloc, r_in=2.5, r_out=3.5)
        annulus_masks = annulus_aperture.to_mask(method='center')

    if (Tmag<=13 and Tmag>11):

        aperturemin = CircularAperture(starloc, r=1)
        aperture = CircularAperture(starloc, r=2)
        pixels1=2
        pixels2=3
        pixels3=4
        annulus_aperture = CircularAnnulus(starloc, r_in=3., r_out=4)
        annulus_masks = annulus_aperture.to_mask(method='center')

    if (Tmag<=11 and Tmag>9):

        aperturemin = CircularAperture(starloc, r=1)
        aperture = CircularAperture(starloc, r=2.5)
        pixels1=2.5
        pixels2=3.5
        pixels3=4.5
        annulus_aperture = CircularAnnulus(starloc, r_in=3.5, r_out=4.5)
        annulus_masks = annulus_aperture.to_mask(method='center')

    if (Tmag<=9):

        aperturemin = CircularAperture(starloc, r=1)
        aperture = CircularAperture(starloc, r=3)
        pixels1=3
        pixels2=4
        pixels3=5
        annulus_aperture = CircularAnnulus(starloc, r_in=4., r_out=5)
        annulus_masks = annulus_aperture.to_mask(method='center')

    star_ap_flux_ratio, ap_flux, ap_stars, star_sky_flux_ratio, sky_flux, sky_stars, total_flux_ratio, total_stars = crowdinglevel(ra, dec, pixels1, pixels2, pixels3)

    magnitude=[]
    magnitude_err=[]
    magnitude_min=[]
    magnitude_min_err=[]
    time=[]
    skyring=[]
    skyring_min=[]

    hdu1 = hdulist[0]

    flux1=np.zeros(len(hdu1[1].data))
    print(len(hdu1[1].data))
    flux1_err=np.zeros(len(hdu1[1].data))
    fluxmin=np.zeros(len(hdu1[1].data))
    fluxmin_err=np.zeros(len(hdu1[1].data))
    background=np.zeros(len(hdu1[1].data))
    backgroundmin=np.zeros(len(hdu1[1].data))
    mode_back=np.zeros(len(hdu1[1].data))
    median_back=np.zeros(len(hdu1[1].data))


    for j in range(len(hdu1[1].data)):
        datas=hdu1[1].data['FLUX'][j]
        err=hdu1[1].data['FLUX_ERR'][j]
        annulus_data = annulus_masks[0].multiply(hdu1[1].data['FLUX'][j])
        mask = annulus_masks[0].data
        annulus_data_1d = annulus_data[mask > 0]
        phot_table=aperture_photometry(datas, aperture, error=err)
        phot_min=aperture_photometry(datas, aperturemin, error=err)
        for col in phot_table.colnames:
            phot_table[col].info.format = '%.8g'
            phot_min[col].info.format = '%.8g'
        flux1[j]=phot_table['aperture_sum']
        flux1_err[j]=phot_table['aperture_sum_err']
        fluxmin[j]=phot_min['aperture_sum']
        fluxmin_err[j]=phot_min['aperture_sum_err']
        #mode_back, _= mode(annulus_data_1d)
        median_back= np.median(annulus_data_1d)
        #_, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d)
        #background[j] = mode_back * aperture.area
        #backgroundmin[j] = mode_back * aperturemin.area
        background[j] = median_back * aperture.area
        backgroundmin[j] = median_back * aperturemin.area
    setty=(hdu1[1].data['QUALITY']==0)
    flux1=flux1[setty]
    flux1_err=flux1_err[setty]
    background=background[setty]
    fluxmin=fluxmin[setty]
    fluxmin_err=fluxmin_err[setty]
    backgroundmin=backgroundmin[setty]

    time1 = hdu1[1].data['TIME']
    time1 = time1[setty]

    bkg_sum = background
    bkg_sum_min = backgroundmin
    final_sum = flux1 - bkg_sum
    final_sum_min = fluxmin - bkg_sum_min
    bkgSubFlux = final_sum
    bkgSubFluxmin = final_sum_min

    normal=bkgSubFlux
    normalmin=bkgSubFluxmin
    mag=-2.5*np.log10(np.abs(bkgSubFlux))+20.4402281476
    mag_err=1.09*flux1_err/bkgSubFlux

    mag_min=-2.5*np.log10(np.abs(bkgSubFluxmin))+20.4402281476
    mag_err_min=1.09*fluxmin_err/bkgSubFluxmin
    nflux=mag
    nflux_err=np.sqrt(2)*mag_err

    nflux_min=mag_min
    nflux_err_min=np.sqrt(2)*mag_err_min

    dtime=time1

    keep=(flux1>0) & (background<=np.percentile(background,95))
    normflux= nflux[keep]
    normflux_err= nflux_err[keep]
    normflux_min= nflux_min[keep]
    normflux_err_min= nflux_err_min[keep]
    dtime= dtime[keep]
    normal=normal[keep]
    flux1_err=flux1_err[keep]
    normalmin=normalmin[keep]
    fluxmin_err=fluxmin_err[keep]
    bkg_sum=bkg_sum[keep]
    bkg_sum_min=bkg_sum_min[keep]
    medmag=np.median(normflux)
    apercorr=np.abs(Tmag-medmag)
    if (medmag>Tmag):
        normflux=normflux-apercorr
    if (medmag<Tmag):
        normflux=normflux+apercorr

    medmagmin=np.median(normflux_min)
    apercorrmin=np.abs(Tmag-medmagmin)

    if (medmagmin>Tmag):
        normflux_min=normflux_min-apercorrmin
    if (medmagmin<Tmag):
        normflux_min=normflux_min+apercorrmin
    magnitude.extend(normflux)
    magnitude_err.extend(normflux_err)
    magnitude_min.extend(normflux_min)
    magnitude_min_err.extend(normflux_err_min)
    time.extend(dtime)
    skyring.extend(bkg_sum)
    skyring_min.extend(bkg_sum_min)

    magnitude=np.reshape(magnitude,len(magnitude))
    magnitude_err=np.reshape(magnitude_err,len(magnitude_err))
    time=np.reshape(time,len(time))
    dmin=np.min(time)

    periods = np.logspace(-2, 1.4, num=500)

    #############################################################################################
    periodogram = robust_periodogram(time, magnitude, flux_err=magnitude_err, periods=periods)
    periodogram_curve=interp1d(periodogram['periods'], periodogram['power'])
    zper=periodogram_curve(periods)
    neg_to_pos = (np.diff(zper[1:])<=0) & (np.diff(zper[:-1])>=0)
    maxima = periods[1:-1][neg_to_pos]
    periodogram_inverse=interp1d(periodogram['power'], periodogram['periods'])
    fm=max(periodogram_curve(maxima))
    per=periodogram_inverse(fm)

    #############################################################################################

    dat_dir = os.path.join(BASE_DIR, 'dat')
    os.makedirs(dat_dir, exist_ok=True)
    filename1 = os.path.join(dat_dir, "TIC_%i_S_%i_LC.dat" % (tic, sector))
    np.savetxt(filename1, np.c_[time, magnitude, magnitude_err, flux1[keep], background[keep]], header='time magnitude magnitude_err flux_ap_no_corr flux_annulus')

    frac=np.zeros(int(max(time)/per))
    length=max(time)/per
    for i in range(int(length)):
        keep2 = (time>(i)*per) & (time<=(i+1)*per)
        normf=magnitude[keep2]
        if len(normf)!=0:
            upperF=np.percentile(normf,98)
            lowerF=np.percentile(normf,2)
            frac=upperF-lowerF
    fracFlux=np.nanmedian(frac)
    phases = foldAt(time, per, T0=dmin)
    sortIndi = np.argsort(phases)
    phases = phases[sortIndi]
    pflux = magnitude[sortIndi]
    loc=max(periodogram['power'])-0.1*max(periodogram['power'])
    loc2=np.quantile(pflux,0.95)

    grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)
    plt.subplot(grid[1, 0])
    plt.plot(phases*per, pflux, 'bo', alpha=0.2, ms=4)
    plt.ylim(max(magnitude), min(magnitude))
    plt.text(per/2.5, loc2, r'$\Delta M=%.3f$' % (fracFlux), fontsize=10)
    plt.xlabel('Phase [d]', fontsize = 10)
    plt.ylabel('Tmag', fontsize = 10)
    plt.subplot(grid[1, 1])
    plt.plot(periodogram['periods'], periodogram['power'], 'k-')
    plt.xlabel('Period [d]', fontsize = 10)
    plt.axvline(x=per, color='r', alpha=0.1)
    plt.xscale('log')
    plt.text(0.05, loc, "P=%.2f d" % (per), fontsize=10)

    plt.subplot(grid[0, 2])
    plot_cutout(firstImage)
    plt.scatter(starloc[0,0], starloc[0,1], s = 40, color = 'red')
    aperture.plot(color='white', lw=1)
    annulus_aperture.plot(color='orange', lw=1)

    if skyview_available:
        plt.subplot(grid[1, 2])
        plt.imshow(image_data, cmap='YlGnBu_r')
        plt.grid(False)
        plt.xlabel('Pixels',fontsize = 10)
        plt.xlim(210,0)
        plt.ylim(0,210)
    else:
        plt.subplot(grid[1, 2])
        plt.text(0.5, 0.5, 'NASA SkyView API\nout of service temporarily!',
                 ha='center', va='center', fontsize=9, wrap=True)
        plt.xticks([])
        plt.yticks([])
        plt.box(False)

    plt.title(r"$in=%i, out=%i, \frac{f_{in}}{f_{\star}}=%.3f, \frac{f_{out}}{f_{\star}}=%.3f$" % (ap_stars, sky_stars, star_ap_flux_ratio, star_sky_flux_ratio), fontsize=8)
    plt.subplot(grid[0, :2])
    plt.errorbar(time, magnitude, yerr=magnitude_err, color="royalblue", alpha=0.3, fmt='.', capthick=2, capsize=2, ecolor='k')
    plt.xlabel('Time [d]', fontsize = 10)
    plt.ylabel('Tmag', fontsize = 10)
    plt.xlim(min(time)-1, max(time)+1)
    plt.ylim(max(magnitude), min(magnitude))

    plt.title(u"TIC %i, Tmag=%.3f, Sector %i" % (tic, Tmag, sector), fontsize=15)

    plt.tight_layout()
    img_dir = os.path.join(BASE_DIR, 'img')
    os.makedirs(img_dir, exist_ok=True)
    plt.savefig(os.path.join(img_dir, "TIC_%i_S_%i.png" % (tic, sector)), dpi=300)
    plt.cla()
    plt.close()
    #plt.show()
    f.write("%i %.3f %.6f %.6f %.3f %.3f %.3f %.3f %i\n" % (tic, Tmag, ra, dec, per, fm, fracFlux, total_flux_ratio, sector))
    return per, tic, ra, dec, fracFlux


"""#Runner"""

import traceback
###################################################################################################################

for i in range(len(tic)):
    try:
        Tess_Period(tic[i], Tmag[i], sector[i])
        print(i, tic[i], Tmag[i], sector[i])
    except Exception as e:
        print(f"Error en {i}, TIC {tic[i]}: {e}")
        traceback.print_exc()
f.close()


