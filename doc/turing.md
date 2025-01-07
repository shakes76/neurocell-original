# Turing machines
Alan Turing introduced the concept of Universal Computation (UC) via his Universal Turing Machine (UTM). I have written a separate set of notes on the subject, so here I will recall only the intuitive explanation of UC.

## Computation
A computation is a process that produces work, i.e. parts of a computation produce working (imagine the intermediate results of doing a long divion by hand say). Often working is a result of executing a series of instructions (parallel or serial), known as an algorithm, but computations need not be limited to them. This working may or may not result in any meaningful result. If the working is meaningful, it may be referred to as the output or intermediate output/result. Some computations may be useless and they disappear or are isolated and not visible. Others may be the underlying fundamental building blocks of life.

## Finite State Machine
A node, machine, system or cell that maintains a state or states that interacts with its environment by reading in symbols externally to change these states based on these inputs is a finite state machine (FSM). The cell or machine may have many (transition) rules internally depending on inputs, but essentially this is irrelevant in general. A simple machine of this nature could be a counter modulo some number for example. In hardware, no rules are necesary as the register can just overflow to implement the rules with the register size matching the modulus. This type of machine as no way of manipulating or selecting its input as it cannot keep track of what has been read because it has no memory. But it can compute simple algorithms, process input streams and pass on signals as needed.

## Turing Machine
A node, machine, system or cell that maintains state like a finite state machine that can also manipulate and has memory of its interaction from its environment (i.e. keep track of the working for its computations) is a Turing machine (TM). Just as a finite state machine, the cell or machine may or may not have a set of internal (transition) rules for switching states and manipulating its input and keeping track of working. This depends on whether it fulfills any 'meaningful' purpose, though its purpose is irrelevant to its definition as a Turing machine. Traditional definitions would require a machine has rules to be a Turing machine, but this may change in a dynamical system. For example, a machine of this sort may initially seem not to serve a purpose until it is connected to another stream or machine. Typically though, a Turing machine has an initial purpose or reason to interact with its environment through evolution or its construction via its rules. Thus, the rules tend to be implemented to complete a specific computation or convert the inputs to an output for the machine or cell to exist.

Now with the ability to keep track of the working, this machine is capable of processing any input. We can see this as the input combined with the machine's memory and ability to track working can be thought of as a (ticker) tape that the machine can read and write. This leads us to the traditional definition of the TM as being a FSM together with a tape having a read/write head. Rules then also have head movement involved, but this is just a mechanised form of awareness, so only awareness in rules is required.

## Universal Computation
Universal computation (UC) allows one to program a machine to complete any algorithm without knowing apriori whether that algorithm will finish (or halt) successfully.

It shows that we can create an algorithm for any task, even one that can simulate any other task(s), the so called Universal Turing Machine (UTM). The UTM or UC however does not tell us how and what we can program or construct with our computations. It merely tells us, whatever we want is possible or can be found. There is no guarantee that it will halt or it is the optimal algorithm in any sense.



