from corgisim import scene, instrument, inputs
import pytest
import os
def test_cpgs_loading():
    script_dir = os.path.dirname(__file__) 
    # script_dir = os.getcwd()

    # Test error handling 
    filepath = 'wrong/file/path'
    with pytest.raises(FileNotFoundError) as excinfo:  
        scene_list, optics = inputs.load_cpgs_data(filepath)  
    assert str(excinfo.value) == filepath +" does not exists." 
    
    filepath = 'test/test_data/cpgs_incorrect_type.txt'
    abs_path =  os.path.join(script_dir, '..', filepath)

    with pytest.raises(TypeError) as excinfo:  
        scene_list, optics = inputs.load_cpgs_data(abs_path)  
    assert str(excinfo.value) == abs_path +" is not an xml file." 

    # Test polarization not implemented
    filepath = 'test/test_data/cpgs_default.xml'
    abs_path =  os.path.join(script_dir, '..', filepath)

    with pytest.raises(NotImplementedError) as excinfo:  
        scene_list, optics = inputs.load_cpgs_data(abs_path)  
    assert str(excinfo.value) == "Only 0/90 deg and 45/135 deg are implemented" 

    # Test object creation 
    filepath = 'test/test_data/cpgs_without_polarization.xml'
    abs_path =  os.path.join(script_dir, '..', filepath)

    scene_list, optics = inputs.load_cpgs_data(abs_path)
    assert isinstance(scene_list[0], scene.Scene)
    assert isinstance(optics, instrument.CorgiOptics)

    # Test correctness of information ?
  
def test_input():
    #test creation
    #For brevity's sake, not all values are tested
    
    input1 = inputs.Input()
    assert isinstance(input1, inputs.Input)
    assert input1.cgi_mode == 'excam'
    assert input1.cor_type == 'hlc'
    assert input1.em_gain == 1000.0
    assert input1.spectral_type == 'G0V'
    assert input1.polaxis == 0

    #Arguments override default
    optics_keywords ={'cor_type':'hlc_band1', 'polaxis':10 }
    emccd_keywords = {'em_gain': 100.0}
    host_star_properties = {'spectral_type':'G1V'}
    
    input2 = inputs.Input(optics_keywords=optics_keywords, emccd_keywords=emccd_keywords, host_star_properties=host_star_properties, cgi_mode = 'excam_efield')
    assert input2.cor_type == 'hlc_band1'
    assert input2.cgi_mode == 'excam_efield'
    assert input2.polaxis == 10
    assert input2.em_gain == 100.0
    assert input2.spectral_type == 'G1V'

    #test dictionnary and individual values are coherents
    assert input2.optics_keywords['cor_type'] == 'hlc_band1'
    assert input2.optics_keywords['polaxis'] == 10
    assert input2.emccd_keywords['em_gain'] == 100.0
    assert input2.host_star_properties['spectral_type'] == 'G1V'

    #Test individual arguments override dictionnary arguments in case of contradiction
    input3 = inputs.Input(optics_keywords=optics_keywords, emccd_keywords=emccd_keywords, host_star_properties=host_star_properties, cor_type='hlc_band4', em_gain =10.0 , spectral_type='A0V')

    assert input3.cor_type == 'hlc_band4'
    assert input3.cgi_mode == 'excam'
    assert input3.polaxis == 10
    assert input3.em_gain == 10.0
    assert input3.spectral_type == 'A0V'

    #test dictionnary and individual values are coherents
    assert input3.optics_keywords['cor_type'] == 'hlc_band4'
    assert input3.optics_keywords['polaxis'] == 10
    assert input3.emccd_keywords['em_gain'] == 10.0
    assert input3.host_star_properties['spectral_type'] == 'A0V'
    
    #test read_only
    with pytest.raises(AttributeError) as excinfo: 
        input3.cor_type = 'hlc'
    assert str(excinfo.value) == "Cannot set attribute cor_type" 
    with pytest.raises(AttributeError) as excinfo: 
        input3._cor_type = 'hlc'
    assert str(excinfo.value) == "Cannot set attribute _cor_type" 
    assert input3.cor_type == 'hlc_band4'

def test_input_from_cpgs():
    script_dir = os.getcwd()
    filepath = 'test/test_data/cpgs_without_polarization.xml'
    abs_path =  os.path.join(script_dir, filepath)

    input = inputs.load_cpgs_data(abs_path, return_input=True)

    # Test that the correct file is used
    assert input.cpgs_file == abs_path

    scene_list, optics = inputs.load_cpgs_data(abs_path)

    scene_input = scene.Scene(input.host_star_properties)
    optics_input =  instrument.CorgiOptics(input.cgi_mode, input.bandpass, optics_keywords=input.optics_keywords, if_quiet=True, integrate_pixels=True)
    
    # Check that the two methods return the same objects       
    for key, val in optics.__dict__.items():
        # No equality operator for synphot objects 
        if type(val).__module__ == 'synphot.spectrum':
            pass
        #Comparing arrays
        elif type(val).__module__ == 'numpy':
            assert (optics_input.__dict__[key] == val).all()
        else:
            # For dictionnaries, we only check that the key that are presents have the same values 
            if key in ['optics_keywords', 'emccd_keywords', 'host_star_properties']:
                for keyword in (optics.__dict__['optics_keywords'].keys() & optics_input.__dict__['optics_keywords'].keys()):
                    assert optics.__dict__[key][keyword] == optics_input.__dict__[key][keyword]
            else:
                assert optics_input.__dict__[key] == val


    for key, val in scene_list[0].__dict__.items():
        # No equality operator for synphot objects 
        if type(val).__module__ == 'synphot.spectrum':
            pass
        #Comparing arrays
        elif type(val).__module__ == 'numpy':
            assert (scene_input.__dict__[key] == val).all()
        else:
            # For dictionnaries, we only check that the key that are presents have the same values 
            if key in ['optics_keywords', 'emccd_keywords', 'host_star_properties']:
                for keyword in (scene_list[0].__dict__['optics_keywords'].keys() & scene_input.__dict__['optics_keywords'].keys()):
                    assert scene_list[0].__dict__[key][keyword] == scene_input.__dict__[key][keyword]
            else:
                assert scene_input.__dict__[key] == val

if __name__ == '__main__':
    test_cpgs_loading()
    test_input()
    test_input_from_cpgs()