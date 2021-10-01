# Catadioptric-Stereo
Project Work for a MSc 3D Vision course offered at ETH Zurich (252-0579-00L) 
Check out **3D_Final_Raport.pdf** for explenations of the Algorithm used.

## Installation

The source code mainly depends on OpenCV, but for completeness we also provide an environment.yaml file. 

## File Structure

All source files are stored in the top level directory:
1. ```demo_*.py```: demo of our implemented functionality adjusted for different use cases.
2. ```parser.py```: parser which is used by all demos
3. ```intrinsics.py```: functionality for calibration with chessboard
4. ```extrinsics.py```: functionality for essential matrix computation
5. ```segmentation.py```: functionality for manual and automatic mirror segmentation
6. ```utils.py```: Utility functions
7. ```FrameIterator.py```: functionality to iterate frames of videos or a multiple images specified with wildcards
8. ```blender.py```: Code by Daniel Thul to compute intrinsics from blender

Sample data is stored in the ```data``` directory. When running the demos, all temporary files will be stored in an automatically generated ```temp``` directory.

## Demos

The demos are created in such a way that all input paths and options are specified in the first lines. If you want to change paths, please do it there.

By default the demos will ask for a manual mirror split, a window with name 'Select Split' will pop up. To run Lukas-Kanade Optical, please activate this with the ```-m``` option.

The demos will use defaut values for the camera intrinsics. To calibrate using chessboard images, please activate this with the ```-i``` option. Note that in this case a manual split for the calibration images is always required


1. To run the blender scene, we recommend using ```python demo_blender.py -i -m```

2. To run the plant scene, we recommend using ```python demo_plant.py```, since there are no optical flow and intrinsics available.

3. To run the real scene, we reccommend using  ```python demo_real.py -i```, since the optical flow does only work on blender data.


When a computation is done and the result is shown, then the window waits for a key to be pressed to continue. 
