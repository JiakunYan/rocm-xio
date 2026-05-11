#!/bin/bash

# Just mode 1
#  python3 /home/dasidler/iris/ext/rocm-xio/examples/sdma-ep-ping-pong/plot_results.py /home/dasidler/iris/ext/rocm-xio/examples/sdma-ep-ping-pong/results --output ping-pong3.png --table
python3 plot_results.py results/ --auto-ylim --modes 1 -o frame1.png

# Modes 1, 2, 6
python3 plot_results.py results/ --auto-ylim --modes 1 2 6 -o frame2.png

# All modes (or specific subset)
python3 plot_results.py results/ --auto-ylim --modes 1 2 6 3 4 -o frame3.png

# Default: all modes
python3 plot_results.py results/ --auto-ylim -o frame4.png

