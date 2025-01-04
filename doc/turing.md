# Turing machines
Alan Turing introduced the concept of Universal Computation (UC) via his Universal Turing Machine (UTM). I have written a separate set of notes on the subject, so here I will recall only the intuitive explanation of UC.

## Finite State Machine
A node, machine, system or cell that maintains a state or states and interacts with its environment by reading in symbols externally to change these states based on this input is a finite state machine. The cell or machine may have many (transition) rules internally depending on inputs, but essentially this is irrelevant for the most part. A simple machine of this nature could be a counter modulo some number for example. In hardware, no rules are necesary as the register can just overflow to implement the rules with the register size matching the modulus. This type of machine as no way of manipulating or selecting its input as it cannot keep track of what has been read because it has no memory.

## Turing Machine
A node, machine, system or cell that maintains state like a finite state machine that can manipulate and/or has memory of its interaction from its environment is a Turing machine. The cell or machine may or may not have a set of internal (transition) rules for switching states and manipulating its input. This depends on whether it fulfills any 'meaningful' purpose. Typically, the machine has a purpose or reason to interact with its environment through evolution or its construction. Thus, the rules tend to be implemented to complete a task or convert the inputs for the machine or cell to exist.

## Universal Computation
Universal computation (UC) allows one to program a machine to complete any algorithm without knowing apriori whether that algorithm will finish (or halt) successfully.

It shows that we can create an algorithm for any task, even one that can simulate any other task(s), the so called Universal Turing Machine (UTM). The UTM or UC however does not tell us how and what we can program or construct with our computations. It merely tells us, whatever we want is possible or can be found. There is no guarantee that it will halt or it is the optimal algorithm in any sense.



