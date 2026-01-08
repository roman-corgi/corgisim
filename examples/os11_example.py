from corgisim.wavefront_estimation import *

if __name__ == '__main__':
    vmag_ref = 2
    vmag_sci = 0
    t_exp = 1 #sec
    t_tot = 20 #sec
    x_off_mas = 300
    y_off_mas = 0
    dmag = 15
    sptype = 'G0V'

    ref_star_properties = {'Vmag': vmag_ref, 'spectral_type': sptype, 'magtype': 'vegamag'}
    target_star_properties = {'Vmag': vmag_sci, 'spectral_type': sptype, 'magtype': 'vegamag'}
    point_source_info = [{
                            'Vmag': vmag_sci + dmag,
                            'magtype': 'vegamag',
                            'position_x': x_off_mas,
                            'position_y': y_off_mas
                        }]

    outdir = "path/to/save/outputs"

    generate_time_series_from_os11(ref_star_properties, target_star_properties, point_source_info, t_exp,
                                   t_tot, outdir, cor_type='hlc', bandpass='1F', polaxis=10, dm_case='3e-8')
