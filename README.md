## Caltech Pedestrian Dataset Converter

This script converts `.seq` files into `.jpg` files, `.vbb` files into `.pkl` files from [Caltech Pedestrian Dataset](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/).

There is a bug in Windows that python will transform `\\annotations` in path into `\x07nnotations`.

Linux is recommended, or you can change `os.path.join()` function into string format to fix this bug directly.

### Requirements
- Python 2.7
- NumPy 1.10.4
- SciPy 0.17.0
