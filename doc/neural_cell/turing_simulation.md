## Turing Machine Simulation
A Turing machine consists of several important components:
* tape
* read/write head (awareness)
* states
* transition table (program)

We can simulate a Turing machine within our cellular machine. Conventional GoL results not withstanding, we can arrange the inputs and outputs in 1D to mimic the configuration on the tape and the cells form the tape. The multi-dimensional form is equivalent to the 1D TM tape and does not offer more computational power.

In our simulation, instead of a conventional program and read/write head for awareness, we have a series of filters that map every possible awareness configuration, while we learn the program from the states embedded within the cells and desired cost function via gradient back-propagation.

I have shown this Turing Simulation (TS) is capable of learning to add two numbers with inputs and outputs in a 1D configuration.
