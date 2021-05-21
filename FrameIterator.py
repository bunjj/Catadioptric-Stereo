import cv2
import glob

class FrameIterator:
    """
    An iterator to iterate over multiple files containing either 
    images other videos. The files are gathered using pattern matching
    read with the universal cv2.VideoCapture

    ...

    Attributes
    ----------
    pathpattern : str
        input path pattern to be matched
    verbose : int
        verbosity (0,1) for standard output

    Methods
    -------
    current_file():
        Returns the path of the current file.
    """
    def __init__(self, pathpattern, verbose=0):
        self.pathpattern = pathpattern

        # list all files matching the pathpattern
        self.files = glob.glob(self.pathpattern)
        if verbose >= 1: print(f'filenames: {self.files}')

        # check that at least one file exists
        self.n_files = len(self.files)
        if self.n_files < 1:
            raise ValueError(f'Could not find any filename matching the path pattern: \'{pathpattern}\'')

    
    def __iter__(self):
        # initialize counter and current VideoCapture 
        self.index = 0
        try: self.cap = cv2.VideoCapture(self.files[self.index])
        except: raise RuntimeError('Opening VideoCapture from \'{self.files[self.index]}\' failed.')
        return self

    def __next__(self):

        # try to read next frame
        try: ret, frame = self.cap.read()
        except: raise RuntimeError('Reading frame from \'{self.files[self.index]}\' failed.')
        
        # return frame if read was sucessfull
        if ret: return frame

        # try to open next VideoCapture if read was unsucessful
        self.index = self.index + 1

        # stop iterating if there are no more files
        if self.index >= self.n_files:
            raise StopIteration

        # initiallize next VideoCapture
        try: self.cap = cv2.VideoCapture(self.files[self.index])
        except: raise RuntimeError('Opening VideoCapture from \'{self.files[self.index]}\' failed.')

        # return first frame of next VideoCapture 
        return self.__next__()


    def current_frame(self):  
        ''' return path and position of the current frame '''
        path = self.files[self.index]
        pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        return f'{path}::{pos}'
