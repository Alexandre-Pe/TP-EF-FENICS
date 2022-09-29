from dolfin import *

# Create mesh and define function space
mesh = UnitSquare(8, 8)
V = FunctionSpace(mesh, 'Lagrange', 1)

# Define initial condition expression g (will also be used as boundary
# condition), and interpolate into initial function u0
alpha = 3.0
beta = 1.2
g = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t',
               alpha=alpha, beta=beta, t=0, degree=2)
u0 = interpolate(g, V)

# Define boundary condition
bc = DirichletBC(V, g, "on_boundary")

# Define timestep and end-time
dt = 0.3
T = 1.8

# Define variational problem for each time-step
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(beta - 2 - 2*alpha)
a = u*v*dx + dt*inner(grad(u), grad(v))*dx
L = (u0 + dt*f)*v*dx

# Assemble once before the time-stepping begins
A = assemble(a)

# Define function for unknown at this time step
u1 = Function(V)

# Run time-loop
t = dt
while t <= T:
    # Assemble right-hand side vector
    b = assemble(L)

    # Update and apply boundary condition
    g.t = t
    bc.apply(A, b)

    # Solve linear system of equations
    solve(A, u1.vector(), b)

    # Update time and previous function
    t += dt
    u0.assign(u1)

plot(u1, title="Approximated solution")
plot(g, mesh=mesh, title="Exact solution")

print "error = ", errornorm(u1, interpolate(g, V))
interactive()
