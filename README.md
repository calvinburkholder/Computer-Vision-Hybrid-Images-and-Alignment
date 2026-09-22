# Computer-Vision-Hybrid-Images-and-Alignment
Creates hybrid images from pairs of faces

Required Libraries:
-__future__/print_function
-builtins/range
-os
-argparse
-numpy
-PIL/Image function
-skimage/color
-matplotlib.pyplot

How to run program:
The programs runs through a parser in the terminal when the program is run.

In the terminal start with: python3 hybrid_image.py

You will then input these arguments:

--diff: 
This selects the difficulty of the shift that occurs which also selects the folder the face you are selecting is in. Your options include: aligned, easy, medium, large

--num1:
This selects the face of the base photo that has not been shifted. Must be input as a three digit number between 001 and 100

--num2:
This selects the face that has been shifted. Must be input as a three digit number between 001 and 100

--low-sd: 
allows for the changing of the Gaussian standard deviation used for the low frequency component. Defaults to 5 if no input.

--high-sd:
allows for the changing of the Gaussian standard deviation used for the high frequency component. Defaults to 5 if no input.

--weight:
allows for the changing of the alpha/weighting between the two frequency components. Defaults to 1 if no input.

--max-shift:
allows for the changing of the max shift the program may search for in alignment. Defaults to pre-set amounts based on difficulty selected: aligned -> 0, easy -> 10, medium -> 25, large -> 40
Messing with this will likely create unaligned images.

--out:
out path of the resulting hybrid image. Defaults to hybrid.png if no input

Expected Input/Output:

Example parser input:

$ python3 hybrid_image.py --diff easy --num1 001 --num2 002 --low-sd 6 --high-sd 4 --weight 2 --max-shift 10 --out test.png

Output:
The output consists of two objects

1. The output image within the file expressed in --out (or in hybrid.png if not expressed). This consists of the low-frequency and high-frequency image seperated, and the hybrid images produced from the two alignment methods at two different scales, big and small.

2. The terminal will spit out a text based summary of the statistics of the difficulty, the two numbered images used, the estimated shift found from both methods, and the outpath of the image.

