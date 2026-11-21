"""Mode and throughput checks, including direct comparison with CGISim v4.1.

The curve comparison requires CGISim to be installed. CorgiSim's runtime
readers use only the bundled data and do not depend on CGISim.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from corgisim import throughput


CURVE_CASES = [('1', 0, 'excam'), ('1a', '2.25', 'excam'),
               ('2', 0, 'excam'), ('2b', '4.75fpam', 'excam'),
               ('3', 0, 'excam'), ('3e', '4.75fsam', 'excam'),
               ('4', 0, 'excam'), ('4c', '4.75fpam', 'excam'),
               ('2', 0, 'spec'), ('3', '4.75fsam', 'spec'),
               ('lowfs', 0, 'lowfs'), ('lowfs', '2.25', 'lowfs')]


MODE_CASES = [
    ('hlc_band1', '1', 'hlc_20190210b', 9.7, 0.575, 7, 0.54625, 0.60375),
    ('hlc', '1a', 'hlc_20190210b', 9.7, 0.575, 5, 0.54627, 0.56573),
    ('hlc_band2', '2', 'hlc_20200617c', 9.7, 0.660, 11, 0.604, 0.716),
    ('spc-spec_band3', '3', 'spc_20200617', 9.4, 0.730, 11, 0.669, 0.791),
    ('spc-wide_band4', '4', 'spc_20200610', 20.4, 0.825, 7, 0.778, 0.872),
    # The wide-field coronagraph allows any defined bandpass via '*'.
    ('spc-wide_band4', '1', 'spc_20200610', 20.4, 0.575, 7, 0.54625, 0.60375),
]

INVALID_WAVELENGTH_RANGES = ['wrong', 'lam700lam600', 'lamnanlam700', 'lam0lam700']


@pytest.mark.parametrize('cor_type,bandpass,name,owa,lam0,nlam,lower,upper', MODE_CASES)
def test_mode_parameters(cor_type, bandpass, name, owa, lam0, nlam, lower, upper):
    mode, bp = throughput.read_mode('excam', cor_type, bandpass)
    assert mode == dict(sampling_um=13.0, lamref_um=0.5,
                        sampling_lamref_div_D=0.5, name=name, owa_lamref=owa)
    assert bp == dict(lam0_um=lam0, nlam=nlam, minlam_um=lower, maxlam_um=upper)


@pytest.mark.parametrize('name,nd,mode', CURVE_CASES)
def test_full_curve_matches_cgisim_v4_1(name, nd, mode):
    import cgisim

    info_dir = cgisim.lib_dir + '/cgisim_info_dir/'
    bp = cgisim.cgisim_read_bandpass(name, info_dir)
    assert throughput.read_bandpass(name) == bp
    step = (bp['maxlam_um'] - bp['minlam_um']) / (bp['nlam'] - 1)
    band = f"lam{(bp['minlam_um'] - step) * 1000}lam{(bp['maxlam_um'] + step) * 1000}"
    reference_wave, reference_values = cgisim.cgisim_roman_throughput(name, band, nd, mode, info_dir)
    wave, values = throughput.roman_throughput(name, band, nd, mode)
    np.testing.assert_array_equal(wave, reference_wave)
    np.testing.assert_allclose(values, reference_values, rtol=1e-13, atol=0)



def test_invalid_and_missing_reference_data(tmp_path):
    with pytest.raises(ValueError, match='Unknown bandpass'):
        throughput.read_bandpass('nonexistent')
    with pytest.raises(ValueError, match='not allowed'):
        throughput.read_mode('excam', 'hlc_band1', '4')
    with pytest.raises(ValueError, match='Unknown bandpass'):
        throughput.read_mode('excam', 'spc-wide_band4', 'nonexistent')
    with pytest.raises(FileNotFoundError):
        throughput.read_mode('excam', 'hlc_band1', '1', info_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        throughput.read_throughput('filters/nonexistent')
    (tmp_path / 'cgisim_bandpasses.txt').write_text('1 = bandpass name\n0.575\n')
    with pytest.raises(ValueError, match='Incomplete bandpass'):
        throughput.read_bandpass('1', info_dir=str(tmp_path))


@pytest.mark.parametrize('bandpass', INVALID_WAVELENGTH_RANGES)
def test_invalid_wavelength_range(bandpass):
    with pytest.raises(ValueError, match='Bandpass must be'):
        throughput.roman_throughput('1', bandpass, 0, 'excam')


def test_invalid_mode_and_nd():
    with pytest.raises(ValueError, match='Unsupported throughput mode'):
        throughput.roman_throughput('1', 'lam540lam610', 0, 'unknown')
    with pytest.raises(ValueError, match='Invalid ND filter'):
        throughput.roman_throughput('1', 'lam540lam610', 1, 'excam')


def test_explicit_data_directory(tmp_path):
    (tmp_path / 'custom.dat').write_text('5000 0.5\n6000 0.75\n')
    wave, values = throughput.read_throughput('custom', info_dir=str(tmp_path))
    np.testing.assert_array_equal(wave, [5000, 6000])
    np.testing.assert_array_equal(values, [0.5, 0.75])


if __name__ == '__main__':
    for case in MODE_CASES:
        test_mode_parameters(*case)
    for case in CURVE_CASES:
        test_full_curve_matches_cgisim_v4_1(*case)
    with TemporaryDirectory() as directory:
        test_invalid_and_missing_reference_data(Path(directory))
    for bandpass in INVALID_WAVELENGTH_RANGES:
        test_invalid_wavelength_range(bandpass)
    test_invalid_mode_and_nd()
    with TemporaryDirectory() as directory:
        test_explicit_data_directory(Path(directory))
    print('All throughput tests passed.')
