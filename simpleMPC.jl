# This script can be used to find the optimal solution (without iterating/LMPC)

using JuMP
using Ipopt
using PyPlot

N = 300
dt = 0.01
# Create Model
mdl = Model(solver = IpoptSolver(print_level=0,max_cpu_time=10.0))#,linear_solver="ma57",print_user_options="yes"))

# Create variables (these are going to be optimized)
@variable( mdl, z[1:(N+1),1:3], start = 0)          # z = s, ey, epsi, v
@variable( mdl, u[1:N,1:2], start = 0)


# Set bounds
z_lb_6s = ones(N+1,1)*[0 -Inf -Inf]                  # lower bounds on states
z_ub_6s = ones(N+1,1)*[1  Inf  Inf]                  # upper bounds
u_lb_6s = ones(N) * [-Inf -Inf]                           # lower bounds on steering
u_ub_6s = ones(N) * [Inf  Inf]                           # upper bounds
for i=1:2
    for j=1:N
        setlowerbound(u[j,i], u_lb_6s[j,i])
        setupperbound(u[j,i], u_ub_6s[j,i])
    end
end
for i=1:2
    for j=1:N+1
        setlowerbound(z[j,i], z_lb_6s[j,i])
        setupperbound(z[j,i], z_ub_6s[j,i])
    end
end

# System dynamics

@NLconstraint(mdl, z[1,2] == z[N+1,2])
@NLconstraint(mdl, z[1,3] == z[N+1,3])
@NLconstraint(mdl, z[1,1] == 0)
@NLconstraint(mdl, z[N+1,1] == 1)
for i=1:N
    @NLconstraint(mdl, z[i+1,1] == z[i,1] + dt*(z[i,2]))
    @NLconstraint(mdl, z[i+1,2] == z[i,2] + dt*(u[i,1] - z[i,2] + z[i,3]))
    @NLconstraint(mdl, z[i+1,3] == z[i,3] + dt*(u[i,2] + z[i,2]))
end

# State cost
# ---------------------------------
@NLexpression(mdl, costZ, sum{(z[i,2]-0.5)^2,i=1:N+1})    # Follow trajectory

# Objective function
@NLobjective(mdl, Min, costZ)

obj_val = zeros(500)
for i=1:230
    N = 230-i
    @NLconstraint(mdl, z[1,2] == z[N+1,2])
    @NLconstraint(mdl, z[1,3] == z[N+1,3])
    @NLconstraint(mdl, z[1,1] == 0)
    @NLconstraint(mdl, z[N+1,1] == 1)
    @NLexpression(mdl, costZ, sum{(z[i,2]-0.5)^2,i=1:N+1})    # Follow trajectory
    @NLobjective(mdl, Min, costZ)

    solve(mdl)
    obj_val[i] = getobjectivevalue(mdl)
    println(obj_val[i])
end

sol_z = getvalue(z)

plot(sol_z[:,1:3],"-o")
grid("on")
legend(["x1","x2","x3"])
figure()
plot(getvalue(u),"-o")
legend(["u1","u2"])
grid("on")
