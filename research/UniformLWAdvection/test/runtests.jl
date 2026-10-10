using Test
using UniformLWAdvection
using Base.Threads

# Deliberately separate, scalar, face-flux reference: one positive-oriented
# flux per periodic face, then a second pass gathers from those fluxes.
function reference_sweep(q, speed, ratio, axis)
    nx, ny = size(q)
    faces = similar(q)
    output = similar(q)
    for j in 1:ny, i in 1:nx
        ip, jp = axis == 1 ? (mod1(i+1,nx),j) : (i,mod1(j+1,ny))
        faces[i,j] = speed[i,j]*(q[i,j]+q[ip,jp])/2 -
                     speed[i,j]^2*ratio*(q[ip,jp]-q[i,j])/2
    end
    for j in 1:ny, i in 1:nx
        im, jm = axis == 1 ? (mod1(i-1,nx),j) : (i,mod1(j-1,ny))
        output[i,j] = q[i,j] - ratio*(faces[i,j]-faces[im,jm])
    end
    output
end

function reference_step(q, u, v, dt)
    nx, ny = size(q)
    a = reference_sweep(q,u,dt*nx/2,1)
    b = reference_sweep(a,v,dt*ny,2)
    reference_sweep(b,u,dt*nx/2,1)
end

# A distinct face-array implementation of the conservative variable-speed
# midpoint predictor, used only by tests.
function variable_reference_sweep(q, speed, ratio, axis)
    nx, ny = size(q)
    faces = similar(q)
    output = similar(q)
    for j in 1:ny, i in 1:nx
        ip, jp = axis == 1 ? (mod1(i+1,nx),j) : (i,mod1(j+1,ny))
        im, jm = axis == 1 ? (mod1(i-1,nx),j) : (i,mod1(j-1,ny))
        leftspeed = (speed[im,jm]+speed[i,j])/2
        rightspeed = (speed[i,j]+speed[ip,jp])/2
        predictor = (q[i,j]+q[ip,jp])/2 -
                    ratio*(rightspeed*q[ip,jp]-leftspeed*q[i,j])/2
        faces[i,j] = speed[i,j]*predictor
    end
    for j in 1:ny, i in 1:nx
        im, jm = axis == 1 ? (mod1(i-1,nx),j) : (i,mod1(j-1,ny))
        output[i,j] = q[i,j] - ratio*(faces[i,j]-faces[im,jm])
    end
    output
end

function variable_reference_step(q,u,v,dt)
    nx, ny = size(q)
    a = variable_reference_sweep(q,u,dt*nx/2,1)
    b = variable_reference_sweep(a,v,dt*ny,2)
    variable_reference_sweep(b,u,dt*nx/2,1)
end

@testset "construction and contract" begin
    @test_throws ArgumentError State(0,4)
    @test_throws ArgumentError State(4,4;u=Inf)
    s = State(11,7)
    @test_throws ArgumentError Workspace(s; flux=:other)
    @test_throws ArgumentError Workspace(s; backend=:other)
    @test_throws ArgumentError Workspace(s; tile=(0,3))
    @test_throws ArgumentError Workspace(s; threshold=-1)
    @test_throws ArgumentError set_velocity!(s,NaN,0)
    old_u, old_v = copy(s.u), copy(s.v)
    @test_throws ErrorException set_velocity!(s,
        (x,y,t)->x > 0.5 ? error("sample failed") : 0.2,
        (x,y,t)->0.1,0.0)
    @test s.u == old_u && s.v == old_v && s.constant_carrier
    @test_throws ArgumentError set_velocity!(s,
        (x,y,t)->0.2,(x,y,t)->y > 0.5 ? NaN : 0.1,0.0)
    @test s.u == old_u && s.v == old_v && s.constant_carrier
    @test_throws ArgumentError set_velocity!(s,fill(0.2,11,7),fill(NaN,11,7))
    @test s.u == old_u && s.v == old_v && s.constant_carrier
    @test_throws ArgumentError initialize!(s;t=Inf)
    initialize!(s)
    @test mass(s) ≈ 1 atol=2e-15
    @test stable_timestep(s) ≈ 0.4/(0.7*11)
    w = Workspace(s)
    @test memory_bytes(s,w) == 7*11*7*sizeof(Float64)
    ws = Workspace(s;flux=:stored)
    @test memory_bytes(s,ws) == 8*11*7*sizeof(Float64)
    @test_throws ArgumentError step!(s,w,-1.0)
    @test_throws ArgumentError step!(s,w,2.0)
    @test_throws ArgumentError simulate!(s,w,0.01,-1)
    w.first = s.q
    @test_throws ArgumentError step!(s,w,0.01)
end

@testset "independent face reference, seams, and conservation" begin
    for (nx,ny) in ((1,1),(1,7),(2,3),(7,1),(11,7),(23,17)),
        (u,v) in ((0.0,0.0),(0.7,-0.3),(-0.65,0.35),(0.4,0.2)),
        flux in (:recompute,:stored), backend in (:serial,:threads,:spawn)
        s = State(nx,ny;u,v)
        initialize!(s)
        # A nonsymmetric perturbation exposes face orientation and periodic seams.
        for j in 1:ny, i in 1:nx
            s.q[i,j] += 0.037*sin(1.7i+0.9j)
        end
        initial = copy(s.q)
        dt = min(0.007,stable_timestep(s))
        expected = reference_step(initial,s.u,s.v,dt)
        w = Workspace(s;flux,backend,tile=(4,3),threshold=0,
                      batches_per_worker=3)
        track_workers!(w)
        m0 = mass(s)
        step!(s,w,dt)
        @test maximum(abs.(s.q .- expected)) <= 3e-15
        @test abs(mass(s)-m0) <= 3e-15
        @test !isempty(active_workers(w))
        @test length(active_workers(w)) <= Threads.nthreads()
        track_workers!(w,false)
        @test isempty(active_workers(w))
        # Repeated updates also test reuse of both scratch arrays and the flux.
        for _ in 1:3
            expected = reference_step(expected,s.u,s.v,dt)
            step!(s,w,dt)
        end
        @test maximum(abs.(s.q .- expected)) <= 1.2e-14
    end
end

@testset "stored and recomputed faces agree across tile layouts" begin
    for tile in ((127,1),(7,5),(3,11)), backend in (:serial,:threads,:spawn)
        a = State(33,19;u=-0.6,v=0.25)
        b = State(33,19;u=-0.6,v=0.25)
        initialize!(a)
        initialize!(b)
        wa = Workspace(a;flux=:stored,backend,tile,threshold=0)
        wb = Workspace(b;flux=:recompute,backend,tile,threshold=0)
        simulate!(a,wa,0.004,8)
        simulate!(b,wb,0.004,8)
        @test a.q == b.q
        @test a.t == b.t
    end
end

@testset "constant state, zero flow, signed reversal" begin
    for flux in (:recompute,:stored), backend in (:serial,:threads,:spawn)
        s = State(29,21;u=-0.73,v=0.19)
        fill!(s.q,2.125)
        w = Workspace(s;flux,backend,tile=(6,4),threshold=0)
        simulate!(s,w,0.004,20)
        @test all(==(2.125),s.q)
        set_velocity!(s,0.0,0.0)
        old = copy(s.q)
        simulate!(s,w,0.1,3)
        @test s.q == old
        initialize!(s)
        initial = copy(s.q)
        set_velocity!(s,0.65,-0.3)
        simulate!(s,w,0.003,20)
        set_velocity!(s,-0.65,0.3)
        simulate!(s,w,0.003,20)
        @test maximum(abs.(s.q .- initial)) < 0.009
        @test abs(mass(s)-1) < 6e-15
    end
end

@testset "analytic cell averages and second-order translation" begin
    errors = Float64[]
    for n in (32,64,128)
        s = State(n,n;u=0.67,v=-0.31)
        initialize!(s)
        @test abs(mass(s)-1) < 4e-16
        w = Workspace(s;backend=:serial)
        stop_time = 0.25
        nsteps = ceil(Int,stop_time/stable_timestep(s))
        simulate!(s,w,stop_time/nsteps,nsteps)
        @test abs(s.t-stop_time) < 2e-15
        push!(errors,translation_error(s,stop_time))
        @test abs(mass(s)-1) < 3e-14
    end
    orders = log2.(errors[1:2] ./ errors[2:3])
    @test all(>(1.8), orders)
end

@testset "independent constant-coefficient Fourier factor" begin
    nx, ny = 39, 27
    u, v = -0.56, 0.37
    s = State(nx,ny;u,v)
    amplitude = sinc(1/nx)*sinc(2/ny)
    for j in 1:ny, i in 1:nx
        s.q[i,j] = amplitude*cospi(2(i-0.5)/nx + 4(j-0.5)/ny)
    end
    dt, nsteps = 0.004, 13
    cx, cy = u*dt*nx/2, v*dt*ny
    kx, ky = 2pi/nx, 4pi/ny
    gx = 1 - im*cx*sin(kx) + cx^2*(cos(kx)-1)
    gy = 1 - im*cy*sin(ky) + cy^2*(cos(ky)-1)
    factor = (gx^2*gy)^nsteps
    for flux in (:recompute,:stored), backend in (:serial,:threads,:spawn)
        state = deepcopy(s)
        w = Workspace(state;flux,backend,tile=(8,5),threshold=0)
        simulate!(state,w,dt,nsteps)
        largest = 0.0
        for j in 1:ny, i in 1:nx
            theta = kx*(i-0.5) + ky*(j-0.5)
            largest = max(largest,abs(state.q[i,j]-amplitude*real(factor*cis(theta))))
        end
        @test largest < 4e-15
    end
end

@testset "unlimited LW oscillation is characterized, not clipped" begin
    s = State(64,3;u=0.7,v=0)
    for j in axes(s.q,2), i in axes(s.q,1)
        s.q[i,j] = 17 <= i <= 48 ? 1.0 : 0.0
    end
    m0 = mass(s)
    step!(s,Workspace(s),stable_timestep(s))
    @test minimum(s.q) < 0
    @test maximum(s.q) > 1
    @test abs(mass(s)-m0) < 2e-15
end

@testset "warmed serial path has no grid-sized allocation" begin
    for n in (128,512), flux in (:recompute,:stored)
        s = State(n,n)
        initialize!(s)
        w = Workspace(s;flux)
        step!(s,w,0.0001)
        @test @allocated(step!(s,w,0.0001)) == 0
    end
end

@testset "warmed variable velocity refresh reuses scratch" begin
    uface = (x,y,t) -> 0.3*sinpi(2y)*cospi(2t)
    vface = (x,y,t) -> -0.2*sinpi(2x)*cospi(2t)
    bytes = Int[]
    for n in (128,512)
        s = State(n,n;u=0,v=0)
        w = Workspace(s;carrier=:variable)
        @test memory_bytes(s,w) == 5n^2*sizeof(Float64)
        set_velocity!(s,uface,vface,0.0)
        @test memory_bytes(s,w) == 7n^2*sizeof(Float64)
        push!(bytes,@allocated set_velocity!(s,uface,vface,0.1))
    end
    @test all(==(0),bytes)
end

@testset "variable predictor, staggered faces, and periodic seams" begin
    constant = State(17,11;u=0.3,v=-0.1)
    variable = deepcopy(constant)
    initialize!(constant)
    initialize!(variable)
    step!(constant,Workspace(constant;carrier=:constant),0.005)
    step!(variable,Workspace(variable;carrier=:variable),0.005)
    @test maximum(abs.(constant.q .- variable.q)) < 2e-15
    for (nx,ny) in ((1,7),(2,3),(17,11)), flux in (:recompute,:stored),
        backend in (:serial,:threads,:spawn)
        s = State(nx,ny;u=0,v=0)
        initialize!(s)
        u = [0.12*sinpi(2(i/nx))*cospi(2(j-0.5)/ny) for i in 1:nx, j in 1:ny]
        v = [-0.09*cospi(2(i-0.5)/nx)*sinpi(2j/ny) for i in 1:nx, j in 1:ny]
        set_velocity!(s,u,v)
        @test stable_timestep(s) > 0
        @test_throws ArgumentError translation_error(s)
        @test_throws ArgumentError initialize!(s)
        w = Workspace(s;flux,backend,carrier=:variable,tile=(4,3),threshold=0)
        dt = min(0.003,stable_timestep(s))
        expected = variable_reference_step(s.q,s.u,s.v,dt)
        m0 = mass(s)
        step!(s,w,dt)
        @test maximum(abs.(s.q .- expected)) < 4e-15
        @test abs(mass(s)-m0) < 4e-15
        plain = Workspace(s;carrier=:constant)
        @test_throws ArgumentError step!(s,plain,dt)
    end
end

const GL8_X = (-0.9602898564975363,-0.7966664774136267,
               -0.5255324099163290,-0.1834346424956498,
                0.1834346424956498, 0.5255324099163290,
                0.7966664774136267, 0.9602898564975363)
const GL8_W = (0.1012285362903763,0.2223810344533745,
               0.3137066458778873,0.3626837833783620,
               0.3626837833783620,0.3137066458778873,
               0.2223810344533745,0.1012285362903763)

function shear_exact_average(i,j,n,t)
    x = (i-0.5)/n
    y = (j-0.5)/n
    displacement = 0.6*sin(2pi*t)/(2pi)
    acc = 0.0
    for k in eachindex(GL8_X)
        yq = y + GL8_X[k]/(2n)
        acc += GL8_W[k]*sin(2pi*(x-displacement*sin(2pi*yq)))
    end
    1 + 0.1*sinc(1/n)*acc
end

@testset "manufactured time-dependent shear, midpoint refresh" begin
    errors = Float64[]
    for n in (32,64,128)
        s = State(n,n;u=0,v=0)
        for j in 1:n, i in 1:n
            s.q[i,j] = 1 + 0.2*sinc(1/n)*sinpi(2(i-0.5)/n)
        end
        w = Workspace(s;carrier=:variable,flux=:recompute,backend=:serial)
        final_time = 0.25
        nsteps = ceil(Int,final_time/(0.35/(0.6n)))
        dt = final_time/nsteps
        for _ in 1:nsteps
            set_velocity!(s,(x,y,t)->0.6*cospi(2t)*sinpi(2y),
                            (x,y,t)->0.0,s.t+dt/2)
            step!(s,w,dt)
        end
        err2 = 0.0
        for j in 1:n, i in 1:n
            err2 += (s.q[i,j]-shear_exact_average(i,j,n,final_time))^2
        end
        push!(errors,sqrt(err2/n^2))
        @test abs(mass(s)-1) < 3e-14
    end
    @test all(>(1.8),log2.(errors[1:2] ./ errors[2:3]))
end

function streamfunction_faces(n, amplitude)
    psi(i,j) = amplitude*sinpi(2i/n)*sinpi(2j/n)
    u = [(psi(i,j)-psi(i,j-1))*n for i in 1:n, j in 1:n]
    v = [-(psi(i,j)-psi(i-1,j))*n for i in 1:n, j in 1:n]
    u,v
end

@testset "discretely divergence-free streamfunction and split constant" begin
    defects = Float64[]
    for n in (24,48,96)
        u,v = streamfunction_faces(n,0.05)
        divmax = 0.0
        for j in 1:n, i in 1:n
            div = n*(u[i,j]-u[mod1(i-1,n),j]+v[i,j]-v[i,mod1(j-1,n)])
            divmax = max(divmax,abs(div))
        end
        @test divmax < 2e-13
        s = State(n,n;u=0,v=0)
        fill!(s.q,1.0)
        set_velocity!(s,u,v)
        w = Workspace(s;carrier=:variable,flux=:stored,backend=:threads,
                      tile=(11,7),threshold=0)
        final_time = 0.15
        # Keep the temporal refinement exactly proportional to grid refinement.
        nsteps = 8*(n÷24)
        @test final_time/nsteps <= stable_timestep(s;cfl=0.35)
        simulate!(s,w,final_time/nsteps,nsteps)
        push!(defects,maximum(abs.(s.q .- 1)))
        @test abs(mass(s)-1) < 3e-14
    end
    @test all(>(1.8),log2.(defects[1:2] ./ defects[2:3]))
end

@testset "independent concurrent workspaces" begin
    a = State(47,23;u=0.4,v=-0.2)
    b = State(53,19;u=-0.25,v=0.45)
    initialize!(a)
    initialize!(b)
    ar, br = deepcopy(a), deepcopy(b)
    wa = Workspace(a;backend=:threads,tile=(8,7),threshold=0)
    wb = Workspace(b;backend=:threads,tile=(7,5),threshold=0)
    war = Workspace(ar;backend=:serial)
    wbr = Workspace(br;backend=:serial)
    ta = Threads.@spawn simulate!(a,wa,0.005,12)
    tb = Threads.@spawn simulate!(b,wb,0.005,12)
    fetch(ta)
    fetch(tb)
    simulate!(ar,war,0.005,12)
    simulate!(br,wbr,0.005,12)
    @test a.q == ar.q
    @test b.q == br.q
end
