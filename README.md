This repo contains three files:
simpleMPC.jl finds the optimal solution to a given (repetitive) system and cost function
simpleMPC2.jl simulates a LMPC controller (non-repetitive)
simpleMPC3.jl simulates a repetitive LMPC controller

Branches:
1. master: main working branch with simple linear system
2. wheel-spring: 2-states nonlinear system that simulates a rotating wheel with a tip which is connected to a fixed spring. The objective is to drive the wheel at a constant angular speed (while the spring forces are changing).