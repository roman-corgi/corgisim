from corgisim import scene, instrument, data_loader
import pytest

def test_cpgs_loading():
    # Test error handling 
    filepath = 'wrong/file/path'
    with pytest.raises(FileNotFoundError) as excinfo:  
        scene_list, optics = data_loader.load_cpgs_data(filepath)  
    assert str(excinfo.value) == filepath +" does not exists." 
    
    filepath = 'corgisim/test/test_data/CGI_0000000000000000014_20221004T2359351_L1_.fits'
    with pytest.raises(TypeError) as excinfo:  
        scene_list, optics = data_loader.load_cpgs_data(filepath)  
    assert str(excinfo.value) == filepath +" is not an xml file." 

    # Test object creation 
    filepath = 'corgisim/test/test_data/cpgs_without_polarization.xml'
    scene_list, optics = data_loader.load_cpgs_data(filepath)
    assert isinstance(scene_list[0], scene.Scene)
    assert isinstance(optics, instrument.CorgiOptics)

    # Test polarization not implemented
    filepath = 'corgisim/test/test_data/cpgs_default.xml'
    with pytest.raises(NotImplementedError) as excinfo:  
        scene_list, optics = data_loader.load_cpgs_data(filepath)  
    assert str(excinfo.value) == "Only 0/90 deg and 45/135 deg are implemented" 

    # Test correctness of information ?
  



if __name__ == '__main__':
    test_cpgs_loading()