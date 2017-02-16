# Simple Mans Model

This is the implementation of the simple man's model to calculate the trajectories of electrons in laser fields.

## Newtons Equations

The force on a particle of mass \(m\) and acceleration \(a\) is defined by

\[ ma = F \]

For a particle of charge \(q\) in an time dependent electric field \(E(t)\), that force is equal to

\[ F = qE(t)\]

Hence, one can write the acceleration as

\[ a(t) = \frac{q}{m}E(t)\]

Integrating this equation yields the velocity and the position.

\[ v(t) = \frac{q}{m}\int E(t)dt + v_0\]

\[ x(t) = \frac{q}{m}\int \int E(t)dtdt + v_0t + x_0\]




## ToDo

- [ ] get parameters from list of atoms
- [ ] make a init function for all user inputs
