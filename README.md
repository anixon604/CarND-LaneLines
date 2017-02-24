# Self-Driving Car Engineer Nanodegree
# Computer Vision
## Project: Find Lane Lines

<img src="laneLines_thirdPass.jpg" width="480" alt="Combined Image" />

### Overview

This project uses computer vision techniques such as color selection, grayscaling, Gaussian smoothing, Canny Edge Detection and Hough transform line detection, in order to detect and overlay lines in images and videos (stream of images).

The pipeline I used for processing the lines was:

1. Grayscale
2. Gaussian blur
3. Canny edge Detection
4. Mask a region of interest
5. Define Hough transform
6. Draw lines on image

**Step 1:** Getting setup with Python

To do this project, you will need Python 3 along with the numpy, matplotlib, and OpenCV libraries, as well as Jupyter Notebook installed.

**Step 2:** Installing moviepy  

To install moviepy run:

`>pip install moviepy`  

and check that the install worked:

`>python`  
`>>>import moviepy`  
`>>>`  
(Ctrl-d to exit Python)

**Step 3:** Opening the code in a Jupyter Notebook

Jupyter is an ipython notebook where you can run blocks of code and see results interactively.  All the code for this project is contained in a Jupyter notebook. To start Jupyter in your browser, run the following command at the terminal prompt (be sure you're in your Python 3 environment!):

`> jupyter notebook`

A browser window will appear showing the contents of the current directory.  Click on the file called "P1.ipynb".  Another browser window will appear displaying the notebook.  Follow the instructions in the notebook to complete the project.  
