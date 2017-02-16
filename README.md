# Simple Mans Model

This is the implementation of the simple man's model to calculate the trajectories of electrons in laser fields.

## Newtons Equations

The force on a particle of mass m and acceleration a is defined by

<img src="./Images/Readme/Fma.png" alt="F=ma" height="20px">

For a particle of charge q in an time dependent electric field E(t), that force is equal to

<img src="./Images/Readme/FqE.png" alt="F=qE" height="25px">

Hence, one can write the acceleration as

<img src="./Images/Readme/aqmE.png" alt="a=q/m*E" height="50px">

Integrating this equation yields the velocity and the position.

<img src="./Images/Readme/vqmE.png" alt="v=q/m*E..." height="50px">

<img src="./Images/Readme/xqmE.png" alt="x=q/m*E..." height="50px">



## ToDo

- [ ] get parameters from list of atoms
- [ ] make a init function for all user inputs
