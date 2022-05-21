import sys
import os.path
import pattern
from generator import ImageGenerator
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

path = Path().resolve()
sys.path.insert(0, os.path.join(path, ""))
file_path = os.path.join(path, "exercise_data/")
label_path = os.path.join(path, "Labels.json")

if __name__ == '__main__':
    
    checker = pattern.Checker(32, 4)
    checker.draw()
    checker.show()
    
    circle = pattern.Circle(256, 32, (128, 128))
    circle.draw()
    circle.show()

    spectrum = pattern.Spectrum(128)
    spectrum.draw()
    spectrum.show()


    gen1 = ImageGenerator(file_path, label_path, 12, (64, 64), rotation=True, mirroring=True, shuffle=True)
    gen1.show()
