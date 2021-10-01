import argparse

def make_parser():
    text = "This is a program to compute the disparity map of a catadioptric image."
    parser = argparse.ArgumentParser(description=text)

    #add argument for intrinsic calibration
    #TODO: adjust implementation for intrinsics! respectively add it
    parser.add_argument("-i", "--intrinsic",
                        help = "compute the intrinsic parameters from chessboard images,"
                            "otherwise they will be loaded form a NP file",
                        action = "store_true")
    
    #add parameter for mirror detection
    #can be adjusted to take a path to a video file as well
    #TODO: hand in path for mirror detection
    parser.add_argument("-m", "--mirror",
                        help = "mirror calibration with optical flow,"
                            "if parameter not set, mirror detection must be done manually",
                        action = "store_true")

    return parser