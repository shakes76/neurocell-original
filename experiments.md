# Experiments

## MLP VNNs
Most of the initial work was done in mimicing traditional MLPs (see test_jax_vnn_mlp_*.py), starting with basic toy datasets like wine in scripts with classify and classify_simple in the names. Also included are the full MNIST models, autoencoders and row forward operator tests.

## Shaped VNNs 
A square-layered perceptron (SLP) and circular-layered perceptrons (CLP) were also implemented early on for simple problems, see slp and clp scripts.

## 3D MLPs
The ease at which to generalize the framework to MLPs in nD is demonstrated in the test_jax_vnn_3D_*.py scripts, which present 3D VNN models of MNIST classification and autoencoders.

## TM-like VNN models 
VNN 2D Models with inputs and outputs arrayed as a 1D tape are shown in scripts test_jax_vnn_tm_*.py

## ALU Models
We train a VNN to mimic an ALU (add, minus, logic, XOR etc.) in two different shapes of initial input and outputs.

We use a 2D square/chip type ALU shape with inputs along rows and controls along columns forming half of a square and outputs a certain depth away along rows from the inputs.
```
test_jax_vnn_alu_arithmetic.py
```

Then we use a TM-like shape of the inputs and outputs all along the columns in a tape like configuration but within a large 2D array to allow a 2D architecture to form.
```
test_jax_vnn_alu_tm_arithmetic.py
```

## TM Tape Model
The following script runs the Turing machine (TM) implementation of a tape based model that runs purely on a single row.
```
test_jax_tm_arithmetic.py
```
It shows that a set of instructions learnt as kernels of the cellular machine can be trained to do arithmetic (usually binary addition) on a model shaped like a tape, i.e. a 1D array of cells. This model uses a single row based forward operator and the UniverseTM class.
