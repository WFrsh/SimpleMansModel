# Simple Mans Model

This is the implementation of the simple man's model to calculate the trajectories of electrons in laser fields.

## Newtons Equations

The force on a particle of mass m and acceleration a is defined by

<img src="https://cloud.githubusercontent.com/assets/25739586/23027146/b31ff510-f463-11e6-9ebf-5d0b3721e64e.png" alt="F=ma" height="20px">

For a particle of charge q in an time dependent electric field E(t), that force is equal to

<img src="https://cloud.githubusercontent.com/assets/25739586/23027143/b31efdfe-f463-11e6-9a3a-fdd066883b38.png" alt="F=qE" height="20px">

Hence, one can write the acceleration as

<img src="https://cloud.githubusercontent.com/assets/25739586/23027147/b32031f6-f463-11e6-8e52-8c4e8e80640d.png" alt="a=q/m*E" height="20px">

Integrating this equation yields the velocity and the position.

<img src="https://cloud.githubusercontent.com/assets/25739586/23027145/b31fed90-f463-11e6-8e9c-ca0bc66b0337.png" alt="a=q/m*E" height="20px">

<img src="https://cloud.githubusercontent.com/assets/25739586/23027144/b31f7d92-f463-11e6-8d18-24380c896a1a.png" alt="a=q/m*E" height="20px">



## ToDo

- [ ] get parameters from list of atoms
- [ ] make a init function for all user inputs
