# This script simulates a repetitive LMPC system.
# The last state of one iteration is used as the initial state in the next iteration.
# State dynamics and cost function have to be entered twice: Once for the simulation and once for the MPC controller (JuMP).

using JuMP
using Ipopt
using PyPlot

function f(x,u,dt)
    z = copy(x)
    z[1] = x[1] + dt*x[2]
    z[2] = x[2] + dt*(u-x[2])
    return z
end

function h(x)
    return sum((x-0.2).^2)
end

n_it = 6        # number of iterations
xlim = 1.0      # periodicity (in state 1)
P = [xlim,0]    # periodicity vector
buf = 2000      # buffer for each iteration
n_x = 2         # number of states
n_u = 1         # number of inputs
N = 5           # number of prediction steps (horizon)
dt = 0.01       # time step
termSet_n = 6   # polynomial degree of terminal set approximation
termCost_n = 6  # polynomial degree of terminal cost approximation
x0 = zeros(n_x) # initial state
Qderiv = [0.0,1.0]

termSet_coeff = zeros(termSet_n+1)
termCost_coeff = zeros(termCost_n+1)

x_save = ones(buf,n_x,n_it)*NaN     # buffer for states
u_save = ones(buf,n_u,n_it)*NaN     # buffer for inputs
Q_save = zeros(buf,1,n_it)          # buffer for Q function
c_save = zeros(buf,3,n_it)          # buffer for cost values

# Create model:
m = Model(solver=IpoptSolver(print_level=0))
@variable(m, x[1:(N+1),1:n_x])          # z = s, ey, epsi, v
@variable(m, u[1:N,n_u])
@NLparameter(m, m_x0[i=1:n_x] == 0)
@NLparameter(m, m_termSet_coeff[i=1:termSet_n+1] == termSet_coeff[i])
@NLparameter(m, m_termCost_coeff[i=1:termCost_n+1] == termCost_coeff[i])
@NLexpression(m, m_termSet,  sum{m_termSet_coeff[i]*x[N+1,1]^(termSet_n+1-i),i=1:termSet_n-1}    + m_termSet_coeff[termSet_n]*x[N+1,1]   + m_termSet_coeff[termSet_n+1])
@NLexpression(m, m_termCost, sum{m_termCost_coeff[i]*x[N+1,1]^(termCost_n+1-i),i=1:termCost_n-1} + m_termCost_coeff[termCost_n]*x[N+1,1] + m_termCost_coeff[termCost_n+1])
for i=1:N
    setupperbound(u[i,1], 0.4)
    @constraint(m, x[i+1,1] == x[i,1] + dt*x[i,2])
    @constraint(m, x[i+1,2] == x[i,2] + dt*(u[i,1] - x[i,2]))
end
@NLconstraint(m, [i=1:n_x], x[1,i] == m_x0[i])
#@NLconstraint(m, x[N+1,2] == m_termSet)
@NLexpression(m, costZ, sum{(x[i,2]-0.2)^2* (1-1/(1+e^(100*(1-x[i,1])))),i=1:N})
@NLexpression(m, m_termConstCost, (x[N+1,2] - m_termSet)^2)
@NLexpression(m, derivCost, sum{Qderiv[j]*sum{(x[i,j]-x[i+1,j])^2,i=1:N},j=1:n_x})
@NLobjective(m, Min, costZ + m_termCost + m_termConstCost*1000 + derivCost)
solve(m)

# Create safe set:
j = 1
x_save[1,:,1] = x0
while x_save[j,1,1] <= xlim
    j += 1
    x_save[j,:,1] = f(x_save[j-1,:,1],0.2,dt)
end

x_prev = zeros(n_x)
# Run LMPC
for i=2:n_it
    # Extend saved variables
    x_save[j+1:j+500,:,i-1] = [(1:500)*dt*x_save[j,2,i-1]+x_save[j,1,i-1] ones(500,1)*x_save[j,2,i-1]]
    u_save[j+1:j+500,:,i-1] = ones(500,1)*u_save[j,:,i-1]
    # Create Q function of last iteration
    for k=1:j
        Q_save[k,1,i-1] = h(x_save[k:j,2,i-1])
    end
    x0 = x_save[j,:,i-1]-P'
    x_save[1,:,i] = x0
    println("======================")
    println("x0 = ",x0)
    j = 1
    while x_save[j,1,i] <= xlim
        dist = (x_save[j,1,i]-x_save[:,1,i-1]).^2
        ind = indmin(dist):indmin(dist)+2*N
        # approximate safe set:
        IntMat = zeros(size(ind,1),termSet_n+1)
        for k=1:termSet_n+1
            IntMat[:,k] = x_save[ind,1,i-1].^(termSet_n+1-k)
        end
        termSet_coeff = IntMat\x_save[ind,2,i-1]
        # approximate Q function:
        termCost_coeff = IntMat\Q_save[ind,1,i-1]
        # Set new coefficients and solve MPC problem:
        setvalue(m_termCost_coeff,termCost_coeff)
        setvalue(m_termSet_coeff,termSet_coeff)
        setvalue(m_x0, x_save[j,:,i][:])
        solve(m)
        # Read solution and simulate
        x_sol = getvalue(x)
        println("Interpolation difference: ", x_sol[N+1,1]-x_save[ind[end],1,i-1])
        checkv = 0.0
        for k=1:termSet_n+1
            checkv += termSet_coeff[k]*x_sol[N+1,1].^(termSet_n+1-k)
        end
        println("Terminal constraint difference: ",x_sol[N+1,2]-checkv)
        j += 1
        u_save[j,1,i] = getvalue(u)[1,1]
        x_save[j,:,i] = f(x_save[j-1,:,i],u_save[j,1,i],dt)
        c_save[j,:,i] = [getvalue(costZ) getvalue(m_termCost) getvalue(m_termConstCost)]
        println("One step error = ", x_save[j,:,i]-x_sol[2,:])
        println("u = ", u_save[j,1,i])
        println("cost = ", c_save[j,:,i])
        # Uncomment this to plot prediction and terminal set
        if i>=2 && j <= 10
            subplot(2,1,1)
            plot(x_save[ind,1,i-1],IntMat*termSet_coeff,"-",x_sol[:,1],x_sol[:,2],"-o")
            subplot(2,1,2)
            plot(x_save[ind,1,i-1],IntMat*termCost_coeff)
            readline()
        end
    end
    plot(x_save[:,:,i])
    grid("on")
    readline()
end
plot(x_save[:,2,1])
plot(x_save[:,2,2])
plot(x_save[:,2,3])
grid("on")


# Notes:
# =======
# Even though we're trying to simulate a 'perfect' LMPC, we are using polynomials to approximate the safe set and Q-function.
# Also, we have to use derivative cost on x since we would get non-continuous states otherwise which can't be approximated by polynomials.
