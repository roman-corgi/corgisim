"""Convert disk-model surface brightness to detector count rate."""

import numpy as np
from astropy import units as u
from astropy.constants import c
from synphot import Observation, SourceSpectrum
from synphot.models import ConstFlux1D


def mcfost_to_mjy_arcsec2(
    disk_image,
    wavelength_micron,
    pixel_scale_arcsec,
):
    """
    Convert a MCFOST RT image from its default unit (lambda * F_lambda
    in W m^-2 pixel^-1) to surface brightness in mJy arcsec^-2.

    Parameters
    ----------
    disk_image : ndarray
        2D array of the MCFOST image. Values are assumed to be in units of
        W m^-2 pixel^-1 (lambda * F_lambda per pixel).
    wavelength_micron : float
        Wavelength in microns corresponding to the model (e.g. central
        wavelength of the band).
    pixel_scale_arcsec : float
        Pixel scale on the sky in arcsec / pixel.

    Returns
    -------
    sb_mjy_arcsec2 : astropy.units.Quantity
        2D array of surface brightness in mJy arcsec^-2.

    Notes
    -----
    - MCFOST RT images commonly have units W.m-2.pixel-1 (lambda * F_lambda).
    - We use the relation: F_nu = (lambda * F_lambda) / nu, with
      nu = c / lambda.
    - Surface brightness is then F_nu / pixel_area (per arcsec^2),
      and we convert to mJy arcsec^-2.
    """
    # Attach physical units to the input image: lambda * F_lambda in W / m^2
    img_lambdaFlambda = np.asanyarray(disk_image) * (u.W / u.m**2)

    # Wavelength and frequency
    lam = (wavelength_micron * u.micron).to(u.m)
    nu = (c / lam).to(u.Hz)  # frequency in Hz

    # F_nu per pixel: [W m^-2 Hz^-1] = (lambda * F_lambda) / nu
    fnu_per_pixel = img_lambdaFlambda / nu  # W m^-2 Hz^-1 per pixel

    # Convert F_nu to mJy
    fnu_mjy_per_pixel = fnu_per_pixel.to(u.mJy)

    # Pixel area in arcsec^2
    pixel_area_arcsec2 = (pixel_scale_arcsec * u.arcsec) ** 2  # arcsec^2 / pixel

    # Surface brightness: mJy / arcsec^2
    sb_mjy_arcsec2 = fnu_mjy_per_pixel / pixel_area_arcsec2

    return sb_mjy_arcsec2  # mJy / arcsec^2



def get_disk_countrate_scale(
    surface_brightness_unit,
    bandpass,
    area_cm2,
    pixel_scale_arcsec,
):
    """
    Compute a global scale factor to convert a disk surface brightness map
    in mJy / arcsec^2 to detector count rate (ct / s per pixel).

    This function assumes the input surface brightness unit is mJy/arcsec^2.
    For MCFOST images in W.m^-2.pixel^-1, use disk_to_countrate(), which will
    internally convert the image to mJy/arcsec^2 first and then apply this
    scale.

    Parameters
    ----------
    surface_brightness_unit : str or object convertible to str
        Should describe a surface brightness in mJy/arcsec^2
        (e.g., "mJy/arcsec^2").
    bandpass : synphot.SpectralElement
        Filter or bandpass used in the simulation, same object as for stars
        and planets in corgisim.
    area_cm2 : float or astropy.units.Quantity
        Effective telescope collecting area in cm^2.
    pixel_scale_arcsec : float
        Pixel scale in arcsec per pixel.

    Returns
    -------
    scale : astropy.units.Quantity
        Global scale factor with units:
            ct / s / (mJy / arcsec^2)
        so that:
            countrate_image = sb_image * scale
        where sb_image has unit mJy / arcsec^2.
    """
    unit_str = str(surface_brightness_unit).lower().replace(" ", "")
    area = area_cm2 if isinstance(area_cm2, u.Quantity) else area_cm2 * u.cm**2

    if "mjy" not in unit_str:
        raise ValueError(
            "get_disk_countrate_scale currently only supports 'mJy/arcsec^2'. "
            "For MCFOST images in W.m^-2.pixel^-1, use disk_to_countrate() below, "
            "which converts to mJy/arcsec^2 internally."
        )

    # Reference spectrum: constant F_nu = 1 Jy
    ref_spec = SourceSpectrum(ConstFlux1D, amplitude=1 * u.Jy)

    # Count rate from a 1 Jy source through this bandpass
    obs = Observation(ref_spec, bandpass)
    cr_per_Jy = obs.countrate(area=area)             # ct / s for 1 Jy
    cr_per_mJy = (cr_per_Jy / 1000.0).to(u.ct / u.s) # ct / s per mJy

    # Pixel solid angle on the sky
    pixel_area = (pixel_scale_arcsec * u.arcsec) ** 2  # arcsec^2 / pixel

    # A uniform SB of 1 mJy/arcsec^2 over one pixel has:
    #   flux_per_pixel = 1 mJy/arcsec^2 * pixel_area
    # so counts_per_pixel = flux_per_pixel * (ct/s per mJy)

    return (cr_per_mJy / u.mJy) * pixel_area  # ct/s per (mJy/arcsec^2)



def disk_to_countrate(
    disk_image,
    surface_brightness_unit,
    bandpass,
    area_cm2,
    pixel_scale_arcsec,
    wavelength_micron=None,
    return_quantity=True,
    input_scene=None,
):
    """
    Convert a 2D disk model to detector count rate per pixel.

    Supported input units:
        * "mJy/arcsec^2"      (surface brightness in F_nu)
        * "W.m-2.pixel-1"     (MCFOST RT image, lambda * F_lambda per pixel)
        * "contrast/pixel"     (contrast = pixel flux / stellar total flux)

    If the input is in the default MCFOST unit ("W.m-2.pixel-1"), this function:
        1) Converts the image to mJy/arcsec^2 using wavelength_micron
           and pixel_scale_arcsec.
        2) Uses get_disk_countrate_scale() to convert that surface brightness
           map to counts/s.


    Parameters
    ----------
    disk_image : ndarray
        2D array with the disk model values.
    surface_brightness_unit : str or object convertible to str
        currently supporting "mJy/arcsec^2", "W.m-2.pixel-1", and contrast/pixel
    bandpass : synphot.SpectralElement
        Bandpass used for the observation, same as for the star/planet
        in corgisim.
    area_cm2 : float
        Telescope collecting area in cm^2.
    pixel_scale_arcsec : float
        Pixel scale in arcsec per pixel.
    wavelength_micron : float, optional
        Wavelength in microns corresponding to the MCFOST model. Required
        if surface_brightness_unit is "W.m-2.pixel-1".
    return_quantity : bool, optional
        If True, return an astropy Quantity with units ct / s.
        If False, return a plain numpy array of floats.
    input_scene : corgisim.scene.Scene, optional
        Scene containing the stellar spectrum required for ``contrast/pixel``.

    Returns
    -------
    disk_counts : Quantity or ndarray
        2D map of counts per second per pixel.
    """
    disk_image = np.asanyarray(disk_image)
    unit_string = str(surface_brightness_unit).lower().replace(" ", "").replace("^", "")

    # ------------------------------------------------------------------
    # Case 1: mJy / arcsec^2 (already surface brightness in F_nu)
    # ------------------------------------------------------------------
    if "mjy" in unit_string:
        disk_surface_brightness = disk_image * (u.mJy / u.arcsec**2)
        scale = get_disk_countrate_scale(
            surface_brightness_unit="mJy/arcsec^2",
            bandpass=bandpass,
            area_cm2=area_cm2,
            pixel_scale_arcsec=pixel_scale_arcsec,
        )
        disk_counts = (disk_surface_brightness * scale).to(u.ct / u.s)

    # ------------------------------------------------------------------
    # Case 2: W.m-2.pixel-1  (MCFOST RT image; lambda * F_lambda per pixel)
    # ------------------------------------------------------------------
    elif "w" in unit_string and "m-2" in unit_string and "pixel-1" in unit_string:
        if wavelength_micron is None:
            raise ValueError(
                "wavelength_micron is required for surface_brightness_unit "
                "'W.m^-2.pixel^-1'."
            )
        # Step 1: convert MCFOST default units -> mJy / arcsec^2
        disk_surface_brightness = mcfost_to_mjy_arcsec2(
            disk_image=disk_image,
            wavelength_micron=wavelength_micron,
            pixel_scale_arcsec=pixel_scale_arcsec,
        )
        # Step 2: convert mJy / arcsec^2 -> counts/s using the same scale
        scale = get_disk_countrate_scale(
            surface_brightness_unit="mJy/arcsec^2",
            bandpass=bandpass,
            area_cm2=area_cm2,
            pixel_scale_arcsec=pixel_scale_arcsec,
        )
        disk_counts = (disk_surface_brightness * scale).to(u.ct / u.s)

    # ------------------------------------------------------------------
    # Case 3: contrast / pixel
    # ------------------------------------------------------------------
    elif "contrast" in unit_string:
        # Convert a 2D disk image in contrast units (contrast per pixel)
        # into detector count rate (ct / s per pixel).
        # The contrast in a given spatial pixel C_ij is defined as:
        #     C_ij = F_ij / F_star
        # where F_ij is the flux in pixel (i, j) and F_star is the total flux
        # of the host star in the same band. If the star's total *count rate*
        # in that band is known (e.g., computed with corgisim from the V-band
        # magnitude and stellar spectrum), then:
        #     counts_ij = C_ij * star_countrate

        if input_scene is None or not hasattr(input_scene, "stellar_spectrum"):
            raise ValueError(
                "input_scene with a stellar_spectrum is required for "
                "surface_brightness_unit 'contrast/pixel'."
            )

        area = area_cm2 if isinstance(area_cm2, u.Quantity) else area_cm2 * u.cm**2
        stellar_count_rate = Observation(
            input_scene.stellar_spectrum, bandpass
        ).countrate(area=area)
        disk_counts = disk_image * stellar_count_rate

    else:
        raise ValueError(
            f"Unsupported surface_brightness_unit '{surface_brightness_unit}'. "
            "Currently supports 'mJy/arcsec^2', 'W.m-2.pixel-1', and contrast/pixel"
        )


    if return_quantity:
        return disk_counts
    return disk_counts.to_value(u.ct / u.s)
