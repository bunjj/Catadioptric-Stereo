import argparse

def make_parser():
    text = "This is a program to compute the disparity map and depth estimation of an mirror image."
    parser = argparse.ArgumentParser(description=text)

    #add argument for source file
    parser.add_argument("-s", "--source", help = "input file to apply depth estimation on")
    #add argument for target file
    parser.add_argument("-t", "--target", help = "target path to store files in")
    #add argument for intrinsic calibration
    #TODO: adjust implementation for intrinsics! respectively add it
    parser.add_argument("-i", "--inrinsic",
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
    #TODO:adjust implementation such that this is possible
    parser.add_argument("-l", "--load",
                        help = "loads the previous mirror offset from a tmp file",
                        action = "store_true")

    return parser