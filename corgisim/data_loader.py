from corgisim import scene, instrument
import xml.etree.ElementTree as ET
from  xml.etree.ElementTree import ParseError
def load_cpgs_data(filepath):
    """Creates a scene, a detector and optics based on the content of a cpgs file

    :param filepath: path to the input file
    :type filepath: string
    
    :raises [ErrorType]: [ErrorDescription]
    .
    :return: a scene list and optics 

    """
    # Parse the file 
    try: 
        tree = ET.parse(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"{filepath} does not exists.")
    except ParseError: 
        raise TypeError(f"{filepath} is not an xml file.") 


    root = tree.getroot()
    
    # Create a host star and scene for each target
    target_list = root.find('target_list')
    host_star_properties_list = []
    scene_list = []
    for target in target_list.iter('target'):
        Vmag = float(target.find('v_mag').text)
        # Luminosity currently not an attribute of the target in cpgs file
        sptype = target.find('spec_type').text + target.find('sub_type').text 
        # MAG Type not currently not an attribute of the target in cpgs file, using vegamag by default
        magtype = "vegamag"
        host_star_properties = {'Vmag': Vmag, 'spectral_type': sptype, 'magtype':magtype}
        base_scene = scene.Scene(host_star_properties)

        host_star_properties_list.append(host_star_properties)
        scene_list.append(base_scene)

    # Create optics and detector for every visit

    cpgs_input = root.find('cpgs_input')


    # For now, filter can only take two values in cpgs:
    #   1 <-> Band 1 (575 nm)
    #   2 <-> Band 4 (825 nm)
    filter_dict = {'1':'1', '2':'4'}
    filt = cpgs_input.find('filter').text
    bandpass = filter_dict[filt]
    # For now, coronagraph_mask can only take one value in cpgs:
    #   1 <-> hlc
    
    coronograph_mask = cpgs_input.find('coronagraph_mask').text

    match bandpass:
        case '1':
            if coronograph_mask == '1':
                cor_type = 'hlc_band1'
            else:
                raise NotImplementedError("HLC is the only implemented mode")
        case '4':
            if coronograph_mask == '1':
                cor_type = 'hlc_band4'
            else:
                raise NotImplementedError("HLC is the only implemented mode")                

        case _:
            raise NotImplementedError("Only Band 1 and Band 4 have been implemented.")                

    # Polarization
    # Polarimetry is not yet implemented, but the structure is left as to simplify future implementation
    if cpgs_input.find('with_polarization').text == '1' : 
        match cpgs_input.find('wollaston').text :
         # 0/90 deg
            case '1' :
                raise NotImplementedError("Only 0/90 deg and 45/135 deg are implemented")       

            # 45/135 deg
            case '2' :
                raise NotImplementedError("Only 0/90 deg and 45/135 deg are implemented")       

            case _: 
                raise NotImplementedError("Only 0/90 deg and 45/135 deg are implemented")
    else :
        polaxis = 0         

    # Only mode implemented for now
    cgi_mode = 'excam'

    proper_keywords ={'cor_type':cor_type, 'polaxis':polaxis, 'output_dim':201}

    optics = instrument.CorgiOptics(cgi_mode, bandpass, proper_keywords=proper_keywords, if_quiet=True)

                    
    #For each visit, return simulated image and header
    #for visit in root.find('visit_list').findall('cgi_visit'):
    

    return scene_list, optics, 



