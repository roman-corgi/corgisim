import numpy as np
import os
import time
from astropy.io import fits

class Dataset():
    """
    A sequence of data of the same kind. Can be indexed and looped over

    Args:
        frames_or_filepaths (list): list of either filepaths or data objects (e.g., Image class)

    Attributes:
        all_data (np.array): an array with all the data combined together. First dimension is always number of images
        frames (np.array): list of data objects (probably corgidrp.data.Image)
    """
    def __init__(self, frames_or_filepaths):
        """
        Args:
            frames_or_filepaths (list): list of either filepaths or data objects (e.g., Image class)
        """
        if len(frames_or_filepaths) == 0:
            raise ValueError("Empty list passed in")

        if isinstance(frames_or_filepaths[0], str):
            # list of filepaths
            # TODO: do some auto detection of the filetype, but for now assume it is an image file
            self.frames = []
            for filepath in frames_or_filepaths:
                self.frames.append(Image(filepath))
        else:
            # list of frames
            self.frames = frames_or_filepaths

        # turn lists into np.array for indexing behavior
        if isinstance(self.frames, list):
            self.frames = np.array(self.frames) # list of objects

        # create 3-D cube of all the data
        self.all_data = np.array([frame.data for frame in self.frames])
        # do a clever thing to point all the individual frames to the data in this cube
        # this way editing a single frame will also edit the entire datacube
        for i, frame in enumerate(self.frames):
            frame.data = self.all_data[i]

    def __iter__(self):
        return self.frames.__iter__()

    def __getitem__(self, indices):
        if isinstance(indices, int):
            # return a single element of the data
            return self.frames[indices]
        else:
            # return a subset of the dataset
            return Dataset(self.frames[indices])

    def __len__(self):
        return len(self.frames)

    def save(self, filedir, filenames=None):
        """
        Save each file of data in this dataset into directory

        Args:
            filedir (str): directory to save the files
            filenames (list): a list of output filenames for each file

        """
        # if filenames are not passed, use the default ones
        if filenames is None:
            filenames = []
            for frame in self.frames:
                filename = frame.filename
                filenames.append(frame.filename)

        for filename, frame in zip(filenames, self.frames):
            frame.save(filename=filename, filedir=filedir)

    def update_after_processing_step(self, history_entry, new_all_data=None, header_entries = None):
        """
        Updates the dataset after going through a processing step

        Args:
            history_entry (str): a description of what processing was done. Mention reference files used.
            new_all_data (np.array): (optional) Array of new data. Needs to be the same shape as `all_data`
            header_entries (dict): (optional) a dictionary {} of ext_hdr and err_hdr entries to add or update
        """
        # update data if necessary
        if new_all_data is not None:
            if new_all_data.shape != self.all_data.shape:
                raise ValueError("The shape of new_all_data is {0}, whereas we are expecting {1}".format(new_all_data.shape, self.all_data.shape))
            self.all_data[:] = new_all_data # specific operation overwrites the existing data rather than changing pointers

        # update history and header entries
        for img in self.frames:
            img.ext_hdr['HISTORY'] = history_entry
            if header_entries:
                for key, value in header_entries.items():
                    img.ext_hdr[key] = value
                    
    def copy(self, copy_data=True):
        """
        Make a copy of this dataset, including all data and headers.
        Data copying can be turned off if you only want to modify the headers
        Headers should always be copied as we should modify them any time we make new edits to the data

        Args:
            copy_data (bool): (optional) whether the data should be copied. Default is True

        Returns:
            corgidrp.data.Dataset: a copy of this dataset
        """
        # there's a smarter way to manage memory, but to keep the API simple, we will avoid it for now
        new_frames = [frame.copy(copy_data=copy_data) for frame in self.frames]
        new_dataset = Dataset(new_frames)

        return new_dataset
    
class Image():
    """
    Base class for 2-D image data. Data can be created by passing in the data/header explicitly, or
    by passing in a filepath to load a FITS file from disk

    Args:
        data_or_filepath (str or np.array): either the filepath to the FITS file to read in OR the 2D image data
        pri_hdr (astropy.io.fits.Header): the primary header (required only if raw 2D data is passed in)
        ext_hdr (astropy.io.fits.Header): the image extension header (required only if raw 2D data is passed in)

    Attributes:
        data (np.array): 2-D data for this Image
        pri_hdr (astropy.io.fits.Header): primary header
        ext_hdr (astropy.io.fits.Header): image extension header. Generally this header will be edited/added to
        filename (str): the filename corresponding to this Image
        filedir (str): the file directory on disk where this image is to be/already saved.
        filepath (str): full path to the file on disk (if it exists)
    """
    def __init__(self, data_or_filepath, pri_hdr=None, ext_hdr=None, input_hdulist = None):
        if isinstance(data_or_filepath, str):
            # a filepath is passed in
            with fits.open(data_or_filepath) as hdulist:
                
                #Pop out the primary header
                self.pri_hdr = hdulist.pop(0).header
                #Pop out the image extension
                first_hdu = hdulist.pop(0)
                self.ext_hdr = first_hdu.header
                self.data = first_hdu.data

                #A list of extensions
                self.hdu_names = [hdu.name for hdu in hdulist]

            # parse the filepath to store the filedir and filename
            filepath_args = data_or_filepath.split(os.path.sep)
            if len(filepath_args) == 1:
                # no directory info in filepath, so current working directory
                self.filedir = "."
                self.filename = filepath_args[0]
            else:
                self.filename = filepath_args[-1]
                self.filedir = os.path.sep.join(filepath_args[:-1])

        else:
            # data has been passed in directly
            # creation of a new file in DRP eyes
            if pri_hdr is None or ext_hdr is None:
                raise ValueError("Missing primary and/or extension headers, because you passed in raw data")
            self.pri_hdr = pri_hdr
            self.ext_hdr = ext_hdr
            self.data = data_or_filepath
            self.filedir = "."
            self.filename = ""

            #Take the input hdulist or make a blank one. 
            if input_hdulist is not None:
                self.hdu_list = input_hdulist
                #Keep track of the names 
                for hdu in input_hdulist:
                    self.hdu_names.append(hdu.name)
            else: 
                self.hdu_list = fits.HDUList()

            
            
            #A list of extensions
            
            # record when this file was created and with which version of the pipeline
            self.ext_hdr.set('SIMVERSN', corgisim.version, "corgidrp version that produced this file")
            self.ext_hdr.set('DRPCTIME', time.Time.now().isot, "When this file was saved")

    # create this field dynamically
    @property
    def filepath(self):
        return os.path.join(self.filedir, self.filename)


    def save(self, filename=None, filedir=None):
        """
        Save file to disk with user specified filepath

        Args:
            filename (str): filepath to save to. Use self.filename if not specified
            filedir (str): filedir to save to. Use self.filedir if not specified
        """
        if filename is not None:
            self.filename = filename
        if filedir is not None:
            self.filedir = filedir

        if len(self.filename) == 0:
            raise ValueError("Output filename is not defined. Please specify!")

        prihdu = fits.PrimaryHDU(header=self.pri_hdr)
        exthdu = fits.ImageHDU(data=self.data, header=self.ext_hdr)
        hdulist = fits.HDUList([prihdu, exthdu])

        hdulist.writeto(self.filepath, overwrite=True)
        hdulist.close()

    def copy(self, copy_data=True):
        """
        Make a copy of this image file. including data and headers.
        Data copying can be turned off if you only want to modify the headers
        Headers should always be copied as we should modify them any time we make new edits to the data

        Args:
            copy_data (bool): (optional) whether the data should be copied. Default is True

        Returns:
            corgidrp.data.Image: a copy of this Image
        """
        if copy_data:
            new_data = np.copy(self.data)
        else:
            new_data = self.data # this is just pointer referencing
        new_img = Image(new_data, pri_hdr=self.pri_hdr.copy(), ext_hdr=self.ext_hdr.copy(),
                        err_hdr = self.err_hdr.copy(), dq_hdr = self.dq_hdr.copy())

        # annoying, but we got to manually update some parameters. Need to keep track of which ones to update
        new_img.filename = self.filename
        new_img.filedir = self.filedir

        # update DRP version tracking
        self.ext_hdr['SIMVERSN'] =  corgisim.version
        self.ext_hdr['DRPCTIME'] =  time.Time.now().isot

        return new_img