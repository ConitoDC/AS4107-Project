#!/usr/bin/env python
# coding: utf-8

import os
import sys
import math
import logging
import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
from astropy.stats import sigma_clip
from photutils.aperture import ApertureStats
from photutils.aperture import aperture_photometry
from photutils.aperture import CircularAperture
from photutils.centroids import centroid_sources
from photutils.centroids import centroid_com
from scipy.ndimage.interpolation import rotate
import matplotlib.pyplot as plt
import galsim


def galaxy_gen(argv, real_img, params, x_pixsize, y_pixsize, output_file):
    """
    - Field 
    - Galaxies are all bulge + disk
    - Galaxies are made with Sersic profiles (with diferent parameters)
    - psf is gaussian
    - Noise is poisson

    """
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger("galaxies")

    # Define some parameters we'll use below.
    # Normally these would be read in from some parameter file.

    pixel_scale = params['pixel_scale']         # arcsec/pixel
    image_size_x = x_pixsize                    # x-size of real image in pixels
    image_size_y = y_pixsize                    # y-size of real image in pixels
    
    # random_seed 
    random_seed = 24783923    

    # Setup full empty image:
    syn_image = output_file # name of full output image
    syn_img = os.path.join(syn_image)
    full_image = galsim.ImageF(image_size_x, image_size_y)
    full_image.setOrigin(0, 0)
    
    # Gal parameters
    bulge_n = params['bulge_n']           # sersic index #0.3 <= n <= 6.2
    bulge_re = params['bulge_eff_rad']    # half light radius in arcsec
    disk_n = 1                            # sersic index 
    disk_r0 = params['disk_scale']        # scale radius in arcsec
    gal_q = params['inclination']         # (axis ratio 0 < q < 1)
    gal_beta = params['sky_angle']        # degrees (position angle on the sky) 
    
    # Components individual fluxes
    bd_ratio = params['bd_ratio']
    disk_flux = params['disk_flux']  
    bulge_flux = disk_flux * bd_ratio  
    frac_flux = params['frac_AGN']
        
    # Define Gal profile
    bulge = galsim.Sersic(bulge_n, flux = bulge_flux, half_light_radius=bulge_re)
    disk = galsim.Sersic(disk_n, flux = disk_flux, scale_radius=disk_r0)
    
    # Create Gal
    gal = galsim.Add([bulge, disk])
    gal_shape = galsim.Shear(q=gal_q, beta=gal_beta*galsim.degrees)
    gal = gal.shear(gal_shape)

    # Define AGN and add to Gal
    AGN_flux = frac_flux * bulge_flux 
    AGN = galsim.DeltaFunction(flux=AGN_flux) 
    logger.info('Synthetic galaxy AGN flux is = %s', str(AGN_flux))
    final_add = galsim.Add(AGN, gal)
        
    # Convolve Gal+AGN with the psf
    psf_fwhm = params['psf_fwhm']
    psf = galsim.Gaussian(fwhm=psf_fwhm)
    final_psf = galsim.Convolve(psf, final_add)

    # Convolucion con psf
    #psf_object = params['psf_object']
    #final_psf = galsim.Convolve(final_add, psf_object)

    # Setup wcs from real image
    header = real_img[1].header
    wcs, origin = galsim.wcs.readFromFitsHeader(header, suppress_warning=True)
    full_image.wcs = wcs

    # Find pixel positions for the stamp
    ra = params['syn_gal_ra']*galsim.degrees
    dec = params['syn_gal_dec']*galsim.degrees
    world_pos = galsim.CelestialCoord(ra, dec)
    image_pos = wcs.toImage(world_pos)
    
    # Account for the fractional part of the position in pixels
    x_nominal = image_pos.x + 0.5
    y_nominal = image_pos.y + 0.5
    ix_nominal = int(math.floor(x_nominal + 0.5))
    iy_nominal = int(math.floor(y_nominal + 0.5))
    dx = x_nominal - ix_nominal
    dy = y_nominal - iy_nominal
    offset = galsim.PositionD(dx, dy)

    # Create stamp
    stamp = final_psf.drawImage(wcs=wcs.local(image_pos), offset=offset)
    #logger.info('Synthetic galaxy final pixel position = %s', str(image_pos))
    #logger.info('Synthetic galaxy final coordinate position = %s, %s', str(world_pos.ra/galsim.degrees), str(world_pos.dec/galsim.degrees))
    # Recenter the stamp at the desired position:
    stamp.setCenter(ix_nominal, iy_nominal)

    # Find the **overlapping** bounds:
    bounds = stamp.bounds & full_image.bounds
    full_image[bounds] += stamp[bounds]

    # Noise. We have to do this step at the end, rather than adding to individual postage stamps, 
    # in order to get the noise level right in the overlap regions between postage stamps.    
    sky_level_pixel = 0
    rng = galsim.BaseDeviate(random_seed)
    noise = galsim.PoissonNoise(rng, sky_level=sky_level_pixel)
    full_image.addNoise(noise) 
    #logger.info('Added noise to final large image')

    # Now write the image to disk.  
    full_image.write(syn_img)
    folder = syn_img.split('/')[1]
    #logger.info('Wrote images to %r', '../' + folder)


def add_galaxy(synthetic, background, x_pixsize, y_pixsize, output_file):

    synth_gal = fits.open(synthetic)
    gal = synth_gal[0].data
    bkg_data = background[1].data
    header = background[1].header

    #put the synthetic galaxy into the background image
    inject = gal + bkg_data
    
    # Now write the image to disk.  
    syn_image = output_file
    syn_img = os.path.join(syn_image)    

    fits.writeto(syn_img, inject, header, overwrite=True)
    synth_gal.close()

    return inject



# takes window of only the real galaxy we're comparing and synthetic galaxy 
def galaxy_cutout(synthetic, background, params):

    syn_ra = params['syn_gal_ra']
    syn_dec = params['syn_gal_dec']
    real_ra = params['real_gal_ra']
    real_dec = params['real_gal_dec']
    
    #open synthetic galaxy file
    synth_gal = fits.open(synthetic)
    gal = synth_gal[0].data

    #extract data from real gal file
    bkg_data = background[1].data
    header = background[1].header
    wcs = WCS(header)

    #put the synthetic galaxy into the background image
    inject = gal + bkg_data
    
    #translate sky coords into pixel coordinates for real and syn galaxy
    syn_pix, syn_piy = wcs.all_world2pix(syn_ra, syn_dec, 1)
    real_pix, real_piy = wcs.all_world2pix(real_ra, real_dec, 1)

    window = 100
    
    #synthetic galaxy cutout
    slx = int(syn_pix) - window
    shx = int(syn_pix) + window
    sly = int(syn_piy) - window
    shy = int(syn_piy) + window
    isynth = inject[sly:shy, slx:shx]
    
    #real galaxy cutout
    rlx = int(real_pix) - window
    rhx = int(real_pix) + window
    rly = int(real_piy) - window
    rhy = int(real_piy) + window
    ogal = background[1].data
    ogal = ogal[rly:rhy, rlx:rhx]

    return isynth, ogal

# creates radial profile of major and minor axis of both galaxies and compares them
def radial_profile(isynth, ogal, angle):
    
    #rotate galaxies to be vertically oriented
    rot_synth = rotate(isynth, angle*-1)
    rot_ogal = rotate(ogal, angle*-1)

    #find pixel positions that maximizes flux in each image
    mino, majo = np.where(rot_ogal == np.max(rot_ogal)) 
    mins, majs = np.where(rot_synth == np.max(rot_synth))

    # plot radial profiles along y axis centered at max in the x-axis
    plt.plot(rot_ogal[mino[0], mino[0]-50:mino[0]+50], label='real')
    plt.plot(rot_synth[mins[0], mins[0]-50:mins[0]+50], label='synthetic', color='green')
    plt.axhline(0, color='0.8')
    plt.axvline(50, color='0.8')
    plt.title('Minor axis')
    plt.legend()
    plt.show()

    # plot radial profiles along x axis centered at max in the y-axis
    plt.plot(rot_ogal[majo[0]-50:majo[0]+50,majo[0]],label='real')
    plt.plot(rot_synth[majs[0]-50:majs[0]+50,majs[0]], label='synthetic', color='green')
    plt.axhline(0, color='0.8')
    plt.axvline(50, color='0.8')
    plt.title('Major axis')
    plt.legend()
    plt.show()
    
    return rot_ogal, rot_synth

# creates a range of flux values
def agn_flux(N, percent):
    low = N * (1 - percent)
    high = N * (1 + percent)
    return np.linspace(low,high,10)

# gets a very rough estimate of galaxy parameters
def galaxy_specs(ra, dec, dradius, bradius, header):
    coords = WCS(header)
    pix, piy = coords.all_world2pix(ra, dec, 1)
    
    half = 0.5
    scale = 1/(np.exp(1))

    #turn arcsec into pixel
    disk_r = dradius / .27
    bulge_r = bradius / .27

    #calculate flux
    dflux, dfluxerr, dflag = sep.sum_circle(data, [pix], [piy], disk_r)
    bflux, bfluxerr, bflag = sep.sum_circle(data, [pix], [piy], bulge_r)

    #radius in pixels
    bulge_half, flag = sep.flux_radius(data, [pix], [piy], [bulge_r], half, 
                              normflux=bflux, subpix=5)
    disk_scale, flag = sep.flux_radius(data, [pix], [piy], [disk_r], scale,
                              normflux=dflux, subpix=5)
    disk_half, flag = sep.flux_radius(data, [pix], [piy], [disk_r], half, normflux=dflux, subpix=5)

    #radius in arcsecs
    br_half = bulge_half * 0.27
    dr_scale = disk_scale * 0.27
    dr_half = disk_half * .27     
    
    #disk concentration index
    dr_80, flag = sep.flux_radius(data, [pix], [piy], [disk_r], .8, normflux=dflux, subpix=5)
    dr_20, flag = sep.flux_radius(data, [pix], [piy], [disk_r], .2, normflux=dflux, subpix=5)
    dc_index = 5*np.log10(dr_80/dr_20) 
       
    #bulge concentration index
    br_80, flag = sep.flux_radius(data, [pix], [piy], [bulge_r], .8, normflux=bflux, subpix=5)
    br_20, flag = sep.flux_radius(data, [pix], [piy], [bulge_r], .2, normflux=bflux, subpix=5)
    bc_index = 5*np.log10(br_80/br_20)    
       
    #all parameters in a dictionary
    galaxy_specs = dict(bulge_flux = bflux[0], disk_flux = dflux[0] - bflux[0], bulge_half_r = br_half[0], disk_scale_r = dr_scale[0], bulge_concentration = bc_index, disk_concentration = dc_index)

    return galaxy_specs


