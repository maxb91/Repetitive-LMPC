# This script simulates a repetitive LMPC system.
# ================================================
# The last state of one iteration is used as the initial state in the next iteration.
# The safe set is always *only* the previous iteration
# State dynamics and cost function have to be entered twice: Once for the simulation and once for the MPC controller (JuMP).
# ================================================

using JuMP
using Ipopt
using PyPlot

# State dynamics
function f(x,u,dt)
    z = copy(x)
    z[1] = x[1] + dt*x[2]
    z[2] = x[2] + dt*(u[1]-x[2]-(1-x[3])^2)
    z[3] = x[3] + dt*(u[2]*x[2])
    return z
end

# Cost function
function h(x)
    return length(x)
end

n_it = 30            # number of iterations
xlim = 1.0         # periodicity (in state 1)
P = [xlim,0.0,0.0]        # periodicity vector
buf = 100000        # buffer for each iteration
n_x = 3             # number of states
n_u = 2             # number of inputs
N = 20              # number of prediction steps (horizon)
dt = 0.01           # time step
termSet_n = 4       # polynomial degree of terminal set approximation
termCost_n = 4      # polynomial degree of terminal cost approximation
QderivX = [0.0,0.1,0.1] # derivative cost on states
QderivU = [0.0,0.0]     # derivative cost on input
x0 = zeros(n_x)     # initial state

termSet_coeff = zeros(termSet_n+1,2)
termCost_coeff = zeros(termCost_n+1)

x_save = ones(buf,n_x,n_it)*NaN     # buffer for states
u_save = ones(buf,n_u,n_it)*NaN     # buffer for inputs
Q_save = zeros(buf,1,n_it)          # buffer for Q function
c_save = zeros(buf,3,n_it)          # buffer for cost values

# Create model:
m = Model(solver=IpoptSolver(print_level=0))
@variable(m, x[1:(N+1),1:n_x])          # z = s, ey, epsi, v
@variable(m, u[1:N,1:n_u])
@NLparameter(m, m_x0[i=1:n_x] == 0)
@NLparameter(m, m_termSet_coeff[i=1:termSet_n+1,j=1:2] == termSet_coeff[i,j])
@NLparameter(m, m_termCost_coeff[i=1:termCost_n+1] == termCost_coeff[i])
@NLparameter(m, uprev[i=1:n_u] == 0)
@NLexpression(m, m_termSet[j=1:2],  sum{m_termSet_coeff[i,j]*x[N+1,1]^(termSet_n+1-i),i=1:termSet_n-1} + m_termSet_coeff[termSet_n,j]*x[N+1,1]   + m_termSet_coeff[termSet_n+1,j])
@NLexpression(m, m_termCost, sum{m_termCost_coeff[i]*x[N+1,1]^(termCost_n+1-i),i=1:termCost_n-1} + m_termCost_coeff[termCost_n]*x[N+1,1] + m_termCost_coeff[termCost_n+1])
for i=1:N
    setupperbound(u[i,1], 1.0)
    setlowerbound(u[i,1], -1.0)
    setupperbound(u[i,2], 1.0)
    setlowerbound(u[i,2], -1.0)
    @NLconstraint(m, x[i+1,1] == x[i,1] + dt*x[i,2])
    @NLconstraint(m, x[i+1,2] == x[i,2] + dt*(u[i,1] - x[i,2] - (1-x[i,3])^2))
    @NLconstraint(m, x[i+1,3] == x[i,3] + dt*(u[i,2]*x[i,2]))
end
@NLconstraint(m, [i=1:n_x], x[1,i] == m_x0[i])                                      # initial condition
#@NLconstraint(m, x[N+1,2] == m_termSet)                                            # hard terminal constraint
#@NLexpression(m, costZ, sum{(x[i,2]-0.5)^2* (1-1/(1+e^(100*(xlim-x[i,1])))),i=1:N}) # stage cost (with h(x_F)=0)
@NLexpression(m, costZ, 1)#sum{(x[i,2]-0.5)^2,i=1:N})                                        # stage cost (without h(x_F)=0)
@NLexpression(m, m_termConstCost, sum{(x[N+1,i+1] - m_termSet[i])^2,i=1:2})                         # soft terminal constraint
@NLexpression(m, derivCost, sum{QderivX[j]*sum{(x[i,j]-x[i+1,j])^2,i=1:N},j=1:n_x} + sum{QderivU[j]*((u[1,j]-uprev[j])^2+sum{(u[i,j]-u[i+1,j])^2,i=1:N-1}),j=1:n_u})
@NLobjective(m, Min, costZ + m_termCost + m_termConstCost*1000 + derivCost)         # objective function
solve(m)    # first solve

# Create safe set (= iteration 0, run with constant input):
j = 1
x_save[1,:,1] = x0
while x_save[j,1,1] <= xlim
    j += 1
    x_save[j,:,1] = f(x_save[j-1,:,1],[1.5,0.0],dt)
end

j_prev = 0      # this is a counter that appends the current iteration to the previous iteration safe set
u_prev = [1.0]

# Run LMPC
for i=2:n_it
    # Create Q function of last iteration
    for k=1:j
        Q_save[k,1,i-1] = h(x_save[k:j,2,i-1])
    end
    x0 = x_save[j,:,i-1]-P'
    x_save[1,:,i] = x0
    j_prev = j
    j = 1

    # Start one iteration:
    while x_save[j,1,i] <= xlim
        dist = (x_save[j,1,i]-x_save[:,1,i-1]).^2
        ind = indmin(dist):indmin(dist)+5*N
        # approximate safe set:
        IntMat = zeros(size(ind,1),termSet_n+1)
        for k=1:termSet_n+1
            IntMat[:,k] = x_save[ind,1,i-1].^(termSet_n+1-k)
        end
        termSet_coeff[:,1] = IntMat\x_save[ind,2,i-1]
        termSet_coeff[:,2] = IntMat\x_save[ind,3,i-1]
        # approximate Q function:
        termCost_coeff = IntMat\Q_save[ind,1,i-1]
        # Set new coefficients and solve MPC problem:
        setvalue(m_termCost_coeff,termCost_coeff)
        setvalue(m_termSet_coeff,termSet_coeff)
        setvalue(m_x0, x_save[j,:,i][:])
        solve(m)
        # Read solution and simulate
        x_sol = getvalue(x)
        #println("Interpolation difference: ", x_sol[N+1,1]-x_save[ind[end],1,i-1])  # should be <0 
        checkv = 0.0
        for k=1:termSet_n+1
            checkv += termSet_coeff[k]*x_sol[N+1,1].^(termSet_n+1-k)
        end
        #println("Terminal constraint difference: ",x_sol[N+1,2]-checkv) # should be really close to 0 (soft constraint)
        j += 1
        u_save[j,:,i] = getvalue(u)[1,:]
        x_save[j,:,i] = f(x_save[j-1,:,i],u_save[j,:,i],dt)
        x_save[j_prev+j-1,:,i-1] = x_save[j,:,i]+P'         # append new state to previous safe set (shifted)
        u_save[j_prev+j-1,:,i-1] = u_save[j,:,i]            # append new input to previous safe set
        u_prev = u_save[j,:,i]
        c_save[j,:,i] = [getvalue(costZ) getvalue(m_termCost) getvalue(m_termConstCost)]
        #println("One step error = ", x_save[j,:,i]-x_sol[2,:])      # should be 0 (no model mismatch)
        #println("u = ", u_save[j,1,i])
        println("cost = ", c_save[j,:,i])
        # Uncomment this to plot prediction and terminal set
        # if i>=2 && j <= 10
        #     subplot(2,1,1)
        #     plot(x_save[ind,1,i-1],IntMat*termSet_coeff,"-",x_sol[:,1],x_sol[:,2],"-o")
        #     subplot(2,1,2)
        #     plot(x_save[ind,1,i-1],IntMat*termCost_coeff)
        #     readline()
        # end
    end
    # Finished iteration

    # Plot this iteration
    plot(x_save[:,:,i])
    grid("on")
    #readline()
end

# Finished all iterations, plot results
figure()
for i=1:n_it
    ind = x_save[:,1,i] .<= xlim
    plot(x_save[ind,1,i],x_save[ind,3,i])
end
grid("on")
title("State")

figure()
for i=1:n_it
    ind = x_save[:,1,i] .<= xlim
    plot(x_save[ind,1,i], u_save[ind,:,i])
end
grid("on")
title("Input")

figure()
plot(reshape(Q_save[1,1,:],n_it),"-o")
grid("on")
xlabel("Iteration")
ylabel("J (cost)")
title("Cost")

figure()
plot(reshape(x_save[1,2,:],n_it),"-o")
grid("on")
xlabel("Iteration")
ylabel("x_F")
title("Final state")

figure()
for i=2:n_it
    plot(x_save[:,1,i],c_save[:,1,i]+c_save[:,2,i])
    plot(x_save[:,1,i],c_save[:,1,i],"--")
end
grid("on")

# Notes:
# =======
# Even though we're trying to simulate a 'perfect' LMPC, we are using polynomials to approximate the safe set and Q-function.
# Also, we have to use derivative cost on x since we would get non-continuous states otherwise which can't be approximated by polynomials.
