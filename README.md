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

--method:
This selects the method of alignment that will be used to calculate the estimated shift. Options include: pixel, grad, both. Defaults to both if no input.

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

$ python3 hybrid_image.py --diff easy --num1 001 --num2 002 
--method both --low-sd 6 --high-sd 4 --weight 2 --max-shift 10 --out test.png

Output:
The output consists of two objects

1. The output image within the file expressed in --out (or in hybrid.png if not expressed). This consists of the low-frequency and high-frequency image seperated, and the hybrid image produced from the selected alignment method, or both, at two different scales, big and small.

2. The terminal will spit out a text based summary of the statistics of the difficulty, the two numbered images used, method, the estimated shift found from both methods (none for method not used, if only one), and the outpath of the image.

How to reproduce my main experiments:

Hybrid Image Parameters:

Low-sd:

Image 1 => $ python3 hybrid_image.py --diff aligned --num1 003 --num2 004 --low-sd 2

Image 2 => $ python3 hybrid_image.py --diff aligned --num1 003 --num2 004 --low-sd 5

Image 3 => $ python3 hybrid_image.py --diff aligned --num1 003 --num2 004 --low-sd 10

High-sd:

Image 1 => $ python3 hybrid_image.py --diff aligned --num1 003 --num2 004 --high-sd 2

Image 2 => $ python3 hybrid_image.py --diff aligned --num1 003 --num2 004 --high-sd 5

Image 3 => $ python3 hybrid_image.py --diff aligned --num1 003 --num2 004 --high-sd 10

Weight:

Image 1 => $ python3 hybrid_image.py --diff aligned --num1 003 --num2 004 --weight .5

Image 2 => $ python3 hybrid_image.py --diff aligned --num1 003 --num2 004 --weight 1

Image 3 => $ python3 hybrid_image.py --diff aligned --num1 003 --num2 004 --weight 2

Alignment Comparison:

Easy:

Faces 18/90 => python3 hybrid_image.py --diff easy --num1 018 --num2 090

Faces 72/40 => python3 hybrid_image.py --diff easy --num1 072 --num2 040

Faces 67/76 => python3 hybrid_image.py --diff easy --num1 067 --num2 076

Medium:

Faces 2/67 => python3 hybrid_image.py --diff medium --num1 002 --num2 067

Faces 25/80 => python3 hybrid_image.py --diff medium --num1 025 --num2 080

Faces 75/95 => python3 hybrid_image.py --diff medium --num1 075 --num2 095

Large:

Faces 48/82 => python3 hybrid_image.py --diff large --num1 048 --num2 082

Faces 9/26 => python3 hybrid_image.py --diff large --num1 009 --num2 026

Faces 78/23 => python3 hybrid_image.py --diff large --num1 078 --num2 023

Low-sd:

Image 1 => $ python3 hybrid_image.py --diff easy --num1 018 --num2 090 --low-sd 2

Image 2 => $ python3 hybrid_image.py --diff easy --num1 018 --num2 090 --low-sd 10

High-sd:

Image 1 => $ python3 hybrid_image.py --diff easy --num1 018 --num2 090 --high-sd 2

Image 2 => $ python3 hybrid_image.py --diff easy --num1 018 --num2 090 --high-sd 10

Weight:

Image 1 => $ python3 hybrid_image.py --diff easy --num1 018 --num2 090 --weight .5

Image 2 => $ python3 hybrid_image.py --diff easy --num1 018 --num2 090 --weight 2

Successes, Failures, and Iteration:

Image 1 => $ python3 hybrid_image.py --diff large --num1 001 --num2 002 --method grad

Image 2 => $ python3 hybrid_image.py --diff large --num1 001 --num2 014 --method grad

Image 3 => $ python3 hybrid_image.py --diff large --num1 001 --num2 050 --method grad

For images 1,2,3 I swapped the use of shift_gradient in run_aligned_experiment to shift_nms() to showcase my usage of MSE

Image 4 => $ python3 hybrid_image.py --diff large --num1 001 --num2 050 --method grad

Image 4 is switched back to shift_gradient

Image 5 => $ python3 hybrid_image.py --diff large --num1 001 --num2 050 --method pixel

For Image 5, In run_aligned_experiment, under use_pixel in the shift_image function, I removed the - signs in front of dx_pixel and dy_pixel

Image 6 => $ python3 hybrid_image.py --diff large --num1 001 --num2 050 --method pixel

To change the scale of the small hybrid image in image 6, change small_panel_shrink=0.65 in run_aligned experiment and change to 1

