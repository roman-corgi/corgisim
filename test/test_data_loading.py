from corgisim import scene, instrument, data_loader
import pytest

def test_cpgs_loading():
    filepath = 'wrong/file/path'
    with pytest.raises(FileNotFoundError) as excinfo:  
        scene, optics, detector = load_cpgs_data(filepath)  
    assert str(excinfo.value) == "{filepath} does not exists" 
    
    filepath = 'corgisim/data/CGI_0000000000000000014_20221004T2359351_L1_.fits'
    with pytest.raises(TypeError) as excinfo:  
        scene, optics, detector = load_cpgs_data(filepath)  
    assert str(excinfo.value) == "{filepath} is not an xml file" 
    
    filepath = 'corgisim/data/cpgs_default.xml'
    scene, optics, detector = load_cpgs_data(filepath)
    assert isinstance(base_scene, scene.Scene)
    assert isinstance(optics, instrument.CorgiOptics)
    assert isinstance(detector, instrument.CorgiDetector)
  



if __name__ == '__main__':
    test_cpgs_loading()