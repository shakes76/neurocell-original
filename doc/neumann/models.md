# Models of Self-Replications
John von Neumann proposed 3 models of reproducing machines based around kinematics, cellular structures and excitation-threshold-fatigue (see pg 95-97).

## Fatigue
Jonnie proposed that the neurons in a fatigue system would have internal states to maintain firing and blocking from firing after recent firing, i.e. fatigue (so called refractory period) to mimic real neurons. Real neurons probably do this because of the chemical constraints of building up charge, releasing to fire and building up again. He correctly infers that the internal states would give these neurons the ability to perform as finite state machines.

### Thoughts
However, upon further thought, a fatigue system would be more than just FSMs, which itself is quite a breakthrough itself allowing current artificial neurons more power, but also because it would allow entire sub-networks to behave independently. For example, there could be specialised neurons on the cusp of branches that decide if entire subnets compute or not, or even entire subnets learn to remove themselves from the computation. This would be a similar form to a 'self dropout' learnt by the network! 