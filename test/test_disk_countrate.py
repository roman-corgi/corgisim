"""Tests for converting disk-model pixel values to detector count rate."""

from types import SimpleNamespace

import numpy as np
import pytest
from astropy import units as u
from astropy.constants import c
from synphot import SourceSpectrum, SpectralElement
from synphot.models import Box1D, ConstFlux1D

from corgisim import disk_countrate


def _make_test_bandpass():
    """Create a simple top-hat bandpass centred at 1 micron."""
    return SpectralElement(
        Box1D,
        amplitude=1.0,
        x_0=10000.0,
        width=1000.0,
    )


def test_supported_units_give_same_count_rate():
    """Equivalent fluxes in all supported units should give equal counts."""
    wavelength_micron = 1.0
    pixel_scale_arcsec = 1.0
    area_cm2 = 100.0
    bandpass = _make_test_bandpass()

    relative_brightness = np.array(
        [
            [0.0, 0.5],
            [1.0, 2.0],
        ]
    )

    # At 1 arcsec/pixel, this represents 1 mJy per pixel when the relative
    # brightness is one.
    image_mjy_arcsec2 = relative_brightness

    # Express the same flux as MCFOST lambda * F_lambda in W/m^2/pixel.
    frequency_hz = (
        c / (wavelength_micron * u.micron)
    ).to_value(u.Hz)
    one_mjy_si = (1.0 * u.mJy).to_value(u.W / u.m**2 / u.Hz)
    image_mcfost = relative_brightness * frequency_hz * one_mjy_si

    # Express the same flux as contrast/pixel around a constant 1 Jy star.
    stellar_spectrum = SourceSpectrum(ConstFlux1D, amplitude=1.0 * u.Jy)
    input_scene = SimpleNamespace(stellar_spectrum=stellar_spectrum)
    image_contrast = relative_brightness * 1.0e-3

    converted_surface_brightness = disk_countrate.mcfost_to_mjy_arcsec2(
        disk_image=image_mcfost,
        wavelength_micron=wavelength_micron,
        pixel_scale_arcsec=pixel_scale_arcsec,
    )
    np.testing.assert_allclose(
        converted_surface_brightness.to_value(u.mJy / u.arcsec**2),
        image_mjy_arcsec2,
        rtol=1.0e-12,
        atol=0.0,
    )

    count_rate_mjy = disk_countrate.disk_to_countrate(
        disk_image=image_mjy_arcsec2,
        surface_brightness_unit="mJy/arcsec^2",
        bandpass=bandpass,
        area_cm2=area_cm2,
        pixel_scale_arcsec=pixel_scale_arcsec,
        return_quantity=False,
    )
    count_rate_mcfost = disk_countrate.disk_to_countrate(
        disk_image=image_mcfost,
        surface_brightness_unit="W.m^-2.pixel^-1",
        bandpass=bandpass,
        area_cm2=area_cm2,
        pixel_scale_arcsec=pixel_scale_arcsec,
        wavelength_micron=wavelength_micron,
        return_quantity=False,
    )
    count_rate_contrast = disk_countrate.disk_to_countrate(
        disk_image=image_contrast,
        surface_brightness_unit="contrast/pixel",
        bandpass=bandpass,
        area_cm2=area_cm2,
        pixel_scale_arcsec=pixel_scale_arcsec,
        return_quantity=False,
        input_scene=input_scene,
    )

    assert isinstance(count_rate_mjy, np.ndarray)
    assert isinstance(count_rate_mcfost, np.ndarray)
    assert isinstance(count_rate_contrast, np.ndarray)

    np.testing.assert_allclose(
        count_rate_mcfost,
        count_rate_mjy,
        rtol=1.0e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        count_rate_contrast,
        count_rate_mjy,
        rtol=1.0e-12,
        atol=0.0,
    )


def test_mcfost_input_requires_wavelength():
    """MCFOST lambda * F_lambda input requires its model wavelength."""
    with pytest.raises(ValueError, match="wavelength_micron"):
        disk_countrate.disk_to_countrate(
            disk_image=np.ones((2, 2)),
            surface_brightness_unit="W.m^-2.pixel^-1",
            bandpass=None,
            area_cm2=100.0,
            pixel_scale_arcsec=1.0,
        )


def test_contrast_input_requires_stellar_spectrum():
    """Contrast/pixel input requires a scene containing the host spectrum."""
    with pytest.raises(ValueError, match="stellar_spectrum"):
        disk_countrate.disk_to_countrate(
            disk_image=np.ones((2, 2)),
            surface_brightness_unit="contrast/pixel",
            bandpass=None,
            area_cm2=100.0,
            pixel_scale_arcsec=1.0,
        )


def test_unsupported_surface_brightness_unit():
    """Unrecognized disk-image units should raise a clear error."""
    with pytest.raises(ValueError, match="Unsupported surface_brightness_unit"):
        disk_countrate.disk_to_countrate(
            disk_image=np.ones((2, 2)),
            surface_brightness_unit="Jy/pixel",
            bandpass=None,
            area_cm2=100.0,
            pixel_scale_arcsec=1.0,
        )
