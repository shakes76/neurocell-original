'''
Test load model
'''
import numpy as np
from PySMILI import milxQtImage
# from PySide2 import QtWidgets
from PySide6 import QtWidgets

##create model
N = 1 #unused
# filename = "mlp_classify_cells.npz"
filename = "alu_tm_arithmetic_cells.npz"
# length = None
# num_classes = None
input_size = None
input_offset = None
output_size = None
output_offset = None
# layer_size = None
# layer_offset = None
kernel_size = None

"""
TM ALU file format

weights=params['universe_lp']['w'],
biases=params['universe_lp']['b'],
filters=params['universe_lp']['f'],
states=params['universe_lp']['s'],
N=N, ctrl_bits=ctrl_bits, bit_depth=bit_depth, n=n, input_size=input_size, output_size=output_size,
input_offset_x1=input_offset_x1, input_offset_x2=input_offset_x2,
crtl_size=ctrl_size, ctrl_offset=ctrl_offset, output_offset=output_offset,
kernel_size=kernel_size,
"""

with np.load(filename, allow_pickle=False) as data:
    weights = data['weights']
    biases = data['biases']
    states = data['states']
    filters = data['filters']
    N=data['N']
    # length=data['length']
    # num_classes=data['num_classes']
    input_size=data['input_size']
    output_size=data['output_size']
    # input_offset=data['input_offset']
    input_offset_x1=data['input_offset_x1']
    input_offset_x2=data['input_offset_x2']
    output_offset=data['output_offset']
    ctrl_bits=data['ctrl_bits']
    bit_depth=data['bit_depth']
    n=data['n']
    # layer_size=data['layer_size']
    # layer_offset=data['layer_offset']
    kernel_size=data['kernel_size']
    crtl_size=data['crtl_size']

print("N:", N, "Number of Classes:", num_classes, "kernel_size", kernel_size)
print("input_size:", input_size, "output_size:", output_size, "layer_size:", layer_size)

image_array = np.vstack((weights, biases, states))
print("image array shape:", image_array.shape)

#setup Qt app
app = QtWidgets.QApplication()
mainWindow = QtWidgets.QMainWindow()

app.setOrganizationName("PySMILI")
app.setApplicationName("Image Viewer")

image = milxQtImage(mainWindow)
image.setData(image_array.ravel(), N, N, 3, 1.0)
image.generateImage()
image.setWindowTitle("Image")

mainWindow.setCentralWidget(image)
mainWindow.resize(512, 512)
mainWindow.show()

app.exec()
print("Done")
