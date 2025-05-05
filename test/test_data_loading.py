from corgisim import scene, instrument, data_loader
import pytest
import os
def test_cpgs_loading():
    script_dir = os.path.dirname(__file__) 

    # Test error handling 
    filepath = 'wrong/file/path'
    with pytest.raises(FileNotFoundError) as excinfo:  
        scene_list, optics = data_loader.load_cpgs_data(filepath)  
    assert str(excinfo.value) == filepath +" does not exists." 
    
    print(os.getcwd())
    filepath = 'test_data/CGI_0000000000000000014_20221004T2359351_L1_.fits'
    abs_path =  os.path.join(script_dir, filepath)

    with pytest.raises(TypeError) as excinfo:  
        scene_list, optics = data_loader.load_cpgs_data(abs_path)  
    assert str(excinfo.value) == abs_path +" is not an xml file." 

    # Test polarization not implemented
    filepath = 'test_data/cpgs_default.xml'
    abs_path =  os.path.join(script_dir, filepath)

    with pytest.raises(NotImplementedError) as excinfo:  
        scene_list, optics = data_loader.load_cpgs_data(abs_path)  
    assert str(excinfo.value) == "Only 0/90 deg and 45/135 deg are implemented" 

    # Test object creation 
    filepath = 'test_data/cpgs_without_polarization.xml'
    abs_path =  os.path.join(script_dir, filepath)

    scene_list, optics = data_loader.load_cpgs_data(abs_path)
    assert isinstance(scene_list[0], scene.Scene)
    assert isinstance(optics, instrument.CorgiOptics)
    
    # Test correctness of information ?
  



if __name__ == '__main__':
    test_cpgs_loading()