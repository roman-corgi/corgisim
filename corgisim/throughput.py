"""Functions to read mode parameters and system throughput, originally from CGISim v4.1.

Reference data are bundled in ``data/cgisim_info_dir``. Mode wavelengths are
in microns; throughput-table and returned curve wavelengths are in angstroms.
"""

from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d


def _data_dir(info_dir):
    return Path(info_dir) if info_dir is not None else Path(__file__).parent / 'data' / 'cgisim_info_dir'


def _read_value(stream):
    for line in stream:
        value = line.split('=')[0].strip()
        if value and not value.startswith('#'):
            return value
    raise ValueError('Incomplete bandpass entry in reference data')


def read_bandpass(requested_bandpass, info_dir=None):
    """Read wavelength settings in microns; originally from CGISim v4.1.

    Returns ``lam0_um``, integer ``nlam``, ``minlam_um``, and ``maxlam_um``.
    An unknown bandpass raises ValueError instead of waiting indefinitely.
    """
    with (_data_dir(info_dir) / 'cgisim_bandpasses.txt').open() as stream:
        for line in stream:
            if 'bandpass name' not in line or line.split('=')[0].strip() != requested_bandpass:
                continue
            lam0 = float(_read_value(stream))
            nlam = int(_read_value(stream))
            minlam, maxlam = map(float, _read_value(stream).split())
            return dict(lam0_um=lam0, nlam=nlam, minlam_um=minlam, maxlam_um=maxlam)
    raise ValueError(f'Unknown bandpass: {requested_bandpass}')


def read_mode(cgi_mode, cor_type, requested_bandpass, info_dir=None):
    """Read mode and bandpass dictionaries; originally from CGISim v4.1.

    Arguments and returned dictionary keys match ``cgisim_read_mode``.
    ``info_dir`` defaults to CorgiSim's bundled reference data. Sampling is
    in microns and lambda_ref/D, and the outer working angle is in lambda_ref/D.
    """
    info_dir = _data_dir(info_dir)
    mode_data = dict(sampling_um=0.0, lamref_um=0.0,
                     sampling_lamref_div_D=0.0, name='', owa_lamref=0.0)
    with (info_dir / f'{cgi_mode}.txt').open() as stream:
        for line in stream:
            key = line.strip()
            if key in ('sampling_um', 'lamref_um', 'sampling_lamref_div_D'):
                mode_data[key] = float(_read_value(stream))

    valid_bandpasses = []
    with (info_dir / f'{cor_type}.txt').open() as stream:
        for line in stream:
            key = line.strip()
            if key == 'name':
                mode_data['name'] = _read_value(stream)
            elif key.startswith('owa'):
                mode_data['owa_lamref'] = float(_read_value(stream))
            elif key == 'bandpasses':
                valid_bandpasses = [entry.strip() for entry in stream
                                    if entry.strip() and not entry.lstrip().startswith('#')]

    if requested_bandpass not in valid_bandpasses and '*' not in valid_bandpasses:
        raise ValueError(f'Requested bandpass {requested_bandpass} not allowed for coronagraph type {cor_type}')
    return mode_data, read_bandpass(requested_bandpass, info_dir)


def read_throughput(throughput_file, info_dir=None):
    """Load wavelength (angstrom) and efficiency; originally from CGISim v4.1.

    ``throughput_file`` is a table name without ``.dat``, e.g. ``filters/1``.
    """
    return np.loadtxt(_data_dir(info_dir) / f'{throughput_file}.dat', unpack=True)


def roman_throughput(filter_name, bandpass, nd, mode, info_dir=None):
    """Calculate system throughput; originally from CGISim v4.1.

    Returns wavelength in angstroms and dimensionless throughput, including QE
    but excluding coronagraph-mask losses. ``bandpass`` has the form
    ``lam<minimum_nm>lam<maximum_nm>``. ``nd`` is 0 (none), '2.25',
    '4.75fpam', or '4.75fsam'. The legacy wavelength grid, linear extrapolation,
    nonnegative clipping, and optical-element factors are preserved.
    ``excam_efield`` uses the same throughput as EXCAM; image generation for
    that mode remains unimplemented in CorgiOptics.
    """
    # Numbers of Al, FSS99, uncoated glass, and AR-coated glass elements.
    elements = {'excam': (2, 13, 0, 3), 'excam_efield': (2, 13, 0, 3),
                'lowfs': (2, 9, 0, 4),
                'spec': (3, 12, 0, 6), 'clear': (0, 0, 0, 0)}
    if mode not in elements:
        raise ValueError(f'Unsupported throughput mode: {mode}')
    if nd not in (0, '2.25', '4.75fpam', '4.75fsam'):
        raise ValueError(f'Invalid ND filter: {nd}')
    try:
        prefix, lower, upper = bandpass.split('lam')
        min_lam, max_lam = float(lower) * 10, float(upper) * 10
        if prefix or not np.isfinite([min_lam, max_lam]).all() or min_lam <= 0 or max_lam <= min_lam:
            raise ValueError
    except (ValueError, AttributeError) as error:
        raise ValueError('Bandpass must be lam<minimum_nm>lam<maximum_nm> with 0 < minimum < maximum') from error

    num_al, num_fss99, num_glass, num_ar_glass = elements[mode]
    info_dir = _data_dir(info_dir)
    lam_hrc, reflectance_hrc = read_throughput('hrc_reflectivity', info_dir)
    lam_fss99, reflectance_fss99 = read_throughput('fss99_reflectivity', info_dir)
    lam_al, reflectance_al = read_throughput('al_reflectivity', info_dir)
    lam_filter, transmission_filter = read_throughput('filters/' + filter_name, info_dir)
    ccd_name = 'locam_emccd' if mode == 'lowfs' else 'excam_emccd'
    lam_ccd, qe_ccd = read_throughput(ccd_name, info_dir)
    if nd != 0:
        lam_nd, transmission_nd = read_throughput('filters/nd' + nd, info_dir)
        dnd = np.median(np.diff(lam_nd))
    else:
        lam_nd = np.array([100.0, 5000.0, 10000.0])
        transmission_nd = np.ones(3)
        dnd = lam_nd[1] - lam_nd[0]

    # The 17-point nominal bandpass limits the step; output may have more points.
    lam_bandpass = np.linspace(min_lam, max_lam, 17)
    spacings = [np.median(np.diff(wave)) for wave in
                (lam_ccd, lam_hrc, lam_al, lam_fss99, lam_filter)]
    dlam = min(*spacings, lam_bandpass[1] - lam_bandpass[0], dnd)
    nlam = int((max_lam - min_lam) / dlam + 1)
    lam = np.linspace(min_lam, max_lam, nlam)

    def resample(wave, values):
        return np.clip(interp1d(wave, values, fill_value='extrapolate')(lam), 0, None)

    reflectance_hrc = resample(lam_hrc, reflectance_hrc)
    reflectance_fss99 = resample(lam_fss99, reflectance_fss99)
    reflectance_al = resample(lam_al, reflectance_al)
    qe_ccd = resample(lam_ccd, qe_ccd)
    transmission_filter = resample(lam_filter, transmission_filter)
    transmission_nd = resample(lam_nd, transmission_nd)
    throughput = (qe_ccd * transmission_filter * transmission_nd
                  * reflectance_hrc**7 * reflectance_al**num_al
                  * reflectance_fss99**num_fss99 * 0.99**num_ar_glass * 0.9**num_glass)
    return lam, throughput
