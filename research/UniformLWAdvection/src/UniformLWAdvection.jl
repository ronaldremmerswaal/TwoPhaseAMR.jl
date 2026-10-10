module UniformLWAdvection

using Base.Threads

export State, Workspace, initialize!, set_velocity!, step!, simulate!, mass,
       translation_error, stable_timestep, memory_bytes, track_workers!,
       active_workers

"""
    State(nx, ny; u=0.7, v=-0.3)

Float64 cell averages on the periodic unit square, with `u[i,j]` the right
face speed and `v[i,j]` the upper face speed. There is one stored copy per
physical periodic face. This initial numerical contract supports spatially
constant prescribed carriers; change them with `set_velocity!`. Direct
mutation of the velocity arrays is outside this contract.
"""
mutable struct State
    q::Matrix{Float64}
    u::Matrix{Float64}
    v::Matrix{Float64}
    u_speed::Float64
    v_speed::Float64
    constant_carrier::Bool
    velocity_scratch::Union{Nothing,Tuple{Matrix{Float64},Matrix{Float64}}}
    t::Float64
end

function State(nx::Integer, ny::Integer; u::Real=0.7, v::Real=-0.3)
    nx > 0 && ny > 0 || throw(ArgumentError("grid dimensions must be positive"))
    a, b = Float64(u), Float64(v)
    isfinite(a) && isfinite(b) || throw(ArgumentError("speeds must be finite"))
    State(zeros(nx, ny), fill(a, nx, ny), fill(b, nx, ny), a, b, true,
          nothing, 0.0)
end

abstract type Backend end
struct SerialBackend <: Backend end
struct ThreadsBackend <: Backend end
struct SpawnBackend <: Backend end
struct PolyesterBackend <: Backend end
abstract type Carrier end
struct ConstantCarrier <: Carrier end
struct VariableCarrier <: Carrier end

const Tile = NTuple{4,Int}

"""
    Workspace(state; flux=:recompute, backend=:serial, carrier=:constant,
              tile=(128,32), threshold=16384, batches_per_worker=4)

Two separate intermediate states implement `x(dt/2), y(dt), x(dt/2)`.
`:stored` adds one reusable face-flux array; `:recompute` evaluates each face
twice, with identical orientation and arithmetic. `carrier=:variable` uses the
conservative midpoint predictor with cell speeds averaged from face speeds;
refresh the face velocities at the global step midpoint before calling `step!`.
Backends are `:serial`,
`:threads` (dynamic tile scheduling), `:spawn` (chunk-owned tasks), and
`:polyester` (available after loading the optional Polyester dependency).
Tiles preserve contiguous traversal of the first index. Setting the first
tile dimension to `nx` gives column slabs. Small grids and one-worker
processes use serial execution. A workspace is exclusive to one in-flight
operation; independent state/workspace pairs may run concurrently.
"""
mutable struct Workspace{B<:Backend,F,C<:Carrier}
    first::Matrix{Float64}
    second::Matrix{Float64}
    flux::F
    backend::B
    carrier::C
    tiles::Vector{Tile}
    chunks::Vector{UnitRange{Int}}
    threshold::Int
    record_workers::Bool
    worker_ids::Vector{Int}
end

function Workspace(s::State; flux::Symbol=:recompute, backend::Symbol=:serial,
                   carrier::Symbol=:constant,
                   tile::Tuple{Integer,Integer}=(128,32), threshold::Integer=16384,
                   batches_per_worker::Integer=4)
    flux in (:recompute, :stored) || throw(ArgumentError("unknown flux policy $flux"))
    backend in (:serial, :threads, :spawn, :polyester) ||
        throw(ArgumentError("unknown backend $backend"))
    carrier in (:constant, :variable) || throw(ArgumentError("unknown carrier $carrier"))
    all(>(0), tile) || throw(ArgumentError("tile dimensions must be positive"))
    threshold >= 0 || throw(ArgumentError("threshold must be nonnegative"))
    batches_per_worker > 0 || throw(ArgumentError("batch count must be positive"))
    b = backend === :serial ? SerialBackend() :
        backend === :threads ? ThreadsBackend() :
        backend === :spawn ? SpawnBackend() : PolyesterBackend()
    c = carrier === :constant ? ConstantCarrier() : VariableCarrier()
    nx, ny = size(s.q)
    tiles = Tile[]
    # Tile order and inner traversal both follow Julia's first array index.
    for j in 1:tile[2]:ny, i in 1:tile[1]:nx
        push!(tiles, (i, min(nx,i+tile[1]-1), j, min(ny,j+tile[2]-1)))
    end
    nt = length(tiles)
    nchunks = min(nt, batches_per_worker * Threads.nthreads(:default))
    chunks = [((div((k-1)*nt,nchunks)+1):div(k*nt,nchunks)) for k in 1:nchunks]
    f = flux === :stored ? similar(s.q) : nothing
    Workspace(similar(s.q), similar(s.q), f, b, c, tiles, chunks, Int(threshold),
              false, zeros(Int, 6nt))
end

function set_velocity!(s::State, u::Real, v::Real)
    a, b = Float64(u), Float64(v)
    isfinite(a) && isfinite(b) || throw(ArgumentError("speeds must be finite"))
    fill!(s.u, a)
    fill!(s.v, b)
    s.u_speed, s.v_speed = a, b
    s.constant_carrier = true
    s
end

"""Copy spatially varying face speeds into the state (one value per periodic face)."""
function set_velocity!(s::State, u::AbstractMatrix, v::AbstractMatrix)
    size(s.q) == size(u) == size(v) || throw(DimensionMismatch("face velocity shapes must match q"))
    # Convert and validate both arrays before publishing either. This also
    # handles callers exchanging s.u and s.v without partial copies. The two
    # scratch arrays are allocated on first variable-carrier use only.
    new_u, new_v = velocity_scratch!(s)
    copyto!(new_u,u)
    copyto!(new_v,v)
    all(isfinite,new_u) && all(isfinite,new_v) ||
        throw(ArgumentError("speeds must be finite"))
    copyto!(s.u,new_u)
    copyto!(s.v,new_v)
    s.constant_carrier = false
    s
end

@inline function velocity_scratch!(s::State)
    if s.velocity_scratch === nothing
        s.velocity_scratch = (similar(s.u),similar(s.v))
    end
    return s.velocity_scratch::Tuple{Matrix{Float64},Matrix{Float64}}
end

"""
    set_velocity!(s, uface, vface, tmid)

Sample `uface(x,y,t)` on right faces `(i/nx,(j-1/2)/ny)` and
`vface(x,y,t)` on upper faces `((i-1/2)/nx,j/ny)` at one global midpoint.
Refresh cost belongs outside a `step!` kernel benchmark.
"""
function set_velocity!(s::State, uface::Function, vface::Function, tmid::Real)
    isfinite(tmid) || throw(ArgumentError("midpoint time must be finite"))
    nx, ny = size(s.q)
    new_u, new_v = velocity_scratch!(s)
    for j in 1:ny, i in 1:nx
        a = Float64(uface(i/nx,(j-0.5)/ny,tmid))
        b = Float64(vface((i-0.5)/nx,j/ny,tmid))
        isfinite(a) && isfinite(b) || throw(ArgumentError("speeds must be finite"))
        @inbounds new_u[i,j], new_v[i,j] = a, b
    end
    copyto!(s.u,new_u)
    copyto!(s.v,new_v)
    s.constant_carrier = false
    s
end

# Integral averages, rather than point samples, for the accuracy fixture.
@inline function analytic_average(i, j, nx, ny, t, u, v)
    x = (i-0.5)/nx - u*t
    y = (j-0.5)/ny - v*t
    1.0 + 0.2*sinc(1/nx)*sinpi(2x) + 0.1*sinc(2/ny)*cospi(4y) +
        0.15*sinc(1/nx)*sinc(1/ny)*sinpi(2(x+y))
end

"Initialize a smooth analytic cell-average translation fixture at time `t`."
function initialize!(s::State; t::Real=0.0)
    isfinite(t) || throw(ArgumentError("time must be finite"))
    s.constant_carrier || throw(ArgumentError("analytic translation initialization requires constant speeds"))
    nx, ny = size(s.q)
    for j in 1:ny, i in 1:nx
        @inbounds s.q[i,j] = analytic_average(i,j,nx,ny,t,s.u_speed,s.v_speed)
    end
    s.t = Float64(t)
    s
end

"Conservative starting timestep; returns Inf for a stationary carrier."
function stable_timestep(s::State; cfl::Real=0.4)
    0 < cfl <= 1 || throw(ArgumentError("CFL must lie in (0,1]"))
    nx, ny = size(s.q)
    if s.constant_carrier
        return cfl / max(abs(s.u_speed)*nx, abs(s.v_speed)*ny)
    end
    cfl / max(maximum(abs,s.u)*nx, maximum(abs,s.v)*ny)
end

"Volume-weighted scalar mass on the unit square (serial diagnostic)."
mass(s::State) = sum(s.q)/length(s.q)

"RMS error against analytic constant translation cell averages."
function translation_error(s::State, t::Real=s.t; u::Real=s.u_speed, v::Real=s.v_speed)
    s.constant_carrier || throw(ArgumentError("analytic translation error requires constant speeds"))
    nx, ny = size(s.q)
    total = 0.0
    for j in 1:ny, i in 1:nx
        @inbounds d = s.q[i,j] - analytic_average(i,j,nx,ny,t,u,v)
        total += d*d
    end
    sqrt(total/length(s.q))
end

"Exact bytes in numerical arrays; scheduler metadata is reported separately."
function memory_bytes(s::State, w::Workspace)
    sizeof(s.q)+sizeof(s.u)+sizeof(s.v)+sizeof(w.first)+sizeof(w.second)+
        (w.flux === nothing ? 0 : sizeof(w.flux))+
        (s.velocity_scratch === nothing ? 0 :
            sizeof(s.velocity_scratch[1])+sizeof(s.velocity_scratch[2]))
end

"Enable instrumentation for an untimed step; disable it before benchmarking."
function track_workers!(w::Workspace, enabled::Bool=true)
    fill!(w.worker_ids, 0)
    w.record_workers = enabled
    w
end
active_workers(w::Workspace) = sort!(unique(filter(!iszero, w.worker_ids)))

function foreach_tile!(f, ::SerialBackend, tiles, chunks)
    for t in eachindex(tiles)
        f(t)
    end
    nothing
end

function foreach_tile!(f, ::ThreadsBackend, tiles, chunks)
    Threads.@threads :dynamic for t in eachindex(tiles)
        f(t)
    end
    nothing
end

function foreach_tile!(f, ::SpawnBackend, tiles, chunks)
    @sync for chunk in chunks
        Threads.@spawn for t in chunk
            f(t)
        end
    end
    nothing
end

function foreach_tile!(f, ::Backend, tiles, chunks)
    throw(ArgumentError("load Polyester before selecting backend=:polyester"))
end

@inline function run_tiles!(f, w::Workspace, pass::Int)
    nx, ny = size(w.first)
    serial = w.backend isa SerialBackend || Threads.nthreads(:default) == 1 ||
             nx*ny < w.threshold
    if w.record_workers
        nt = length(w.tiles)
        g = t -> begin
            @inbounds w.worker_ids[t+(pass-1)*nt] = Threads.threadid()
            f(w.tiles[t])
        end
        foreach_tile!(g, serial ? SerialBackend() : w.backend, w.tiles, w.chunks)
    elseif serial
        # The serial baseline does one contiguous traversal, independent of tile
        # granularity, so a candidate cannot win by degrading its one-worker run.
        f((1,nx,1,ny))
    else
        foreach_tile!(t -> f(@inbounds(w.tiles[t])), w.backend, w.tiles, w.chunks)
    end
    nothing
end

# Every face is evaluated in its positive coordinate orientation. Both callers
# of a recomputed face use these exact operations on the same inputs.
@inline lwflux(left, right, speed, ratio) =
    speed * ((left+right)*0.5) - ((speed*speed)*ratio*0.5)*(right-left)

@inline function xcell(q, u, i, j, im, ip, ratio)
    @inbounds qi = q[i,j]
    @inbounds fr = lwflux(qi,q[ip,j],u[i,j],ratio)
    @inbounds fl = lwflux(q[im,j],qi,u[im,j],ratio)
    qi - ratio*(fr-fl)
end

function recompute_tile!(out, q, u, ratio, ::Val{1}, ::ConstantCarrier, tile)
    nx = size(q,1)
    ilo, ihi, jlo, jhi = tile
    for j in jlo:jhi
        # Seam work is outside the vectorizable interior, including one-cell grids.
        if ilo == 1
            @inbounds out[1,j] = xcell(q,u,1,j,nx,min(2,nx),ratio)
        end
        @inbounds @simd for i in max(2,ilo):min(nx-1,ihi)
            out[i,j] = xcell(q,u,i,j,i-1,i+1,ratio)
        end
        if ihi == nx && nx > 1
            @inbounds out[nx,j] = xcell(q,u,nx,j,nx-1,1,ratio)
        end
    end
    nothing
end

function recompute_tile!(out, q, v, ratio, ::Val{2}, ::ConstantCarrier, tile)
    ny = size(q,2)
    ilo, ihi, jlo, jhi = tile
    for j in jlo:jhi
        jm, jp = j == 1 ? ny : j-1, j == ny ? 1 : j+1
        @inbounds @simd for i in ilo:ihi
            qi = q[i,j]
            ft = lwflux(qi,q[i,jp],v[i,j],ratio)
            fb = lwflux(q[i,jm],qi,v[i,jm],ratio)
            out[i,j] = qi - ratio*(ft-fb)
        end
    end
    nothing
end

function flux_tile!(f, q, u, ratio, ::Val{1}, ::ConstantCarrier, tile)
    nx = size(q,1)
    ilo, ihi, jlo, jhi = tile
    for j in jlo:jhi
        @inbounds @simd for i in ilo:min(ihi,nx-1)
            f[i,j] = lwflux(q[i,j],q[i+1,j],u[i,j],ratio)
        end
        if ihi == nx
            @inbounds f[nx,j] = lwflux(q[nx,j],q[1,j],u[nx,j],ratio)
        end
    end
    nothing
end

function flux_tile!(f, q, v, ratio, ::Val{2}, ::ConstantCarrier, tile)
    ny = size(q,2)
    ilo, ihi, jlo, jhi = tile
    for j in jlo:jhi
        jp = j == ny ? 1 : j+1
        @inbounds @simd for i in ilo:ihi
            f[i,j] = lwflux(q[i,j],q[i,jp],v[i,j],ratio)
        end
    end
    nothing
end

# For a variable carrier frozen at the global midpoint, the 1-D Taylor
# predictor at a face is q* = (qL+qR)/2 - τ/(2h)[(a q)R-(a q)L].
# Cell speeds are averages of their two adjacent staggered face speeds.
# Every evaluation of a physical face calls exactly one of these helpers in
# positive orientation; neighboring output cells therefore cancel fluxes.
@inline variable_flux(qleft, qright, face_speed, left_speed, right_speed, ratio) =
    face_speed * ((qleft+qright)*0.5 -
                  ratio*0.5*(right_speed*qright-left_speed*qleft))

@inline function variable_xface(q,u,i,j,ratio,nx)
    im = i == 1 ? nx : i-1
    ip = i == nx ? 1 : i+1
    @inbounds variable_flux(q[i,j],q[ip,j],u[i,j],
                            (u[im,j]+u[i,j])*0.5,
                            (u[i,j]+u[ip,j])*0.5,ratio)
end

@inline function variable_yface(q,v,i,j,ratio,ny)
    jm = j == 1 ? ny : j-1
    jp = j == ny ? 1 : j+1
    @inbounds variable_flux(q[i,j],q[i,jp],v[i,j],
                            (v[i,jm]+v[i,j])*0.5,
                            (v[i,j]+v[i,jp])*0.5,ratio)
end

function recompute_tile!(out,q,u,ratio,::Val{1},::VariableCarrier,tile)
    nx = size(q,1)
    ilo, ihi, jlo, jhi = tile
    for j in jlo:jhi
        @inbounds @simd for i in ilo:ihi
            im = i == 1 ? nx : i-1
            out[i,j] = q[i,j] - ratio*(variable_xface(q,u,i,j,ratio,nx)-
                                       variable_xface(q,u,im,j,ratio,nx))
        end
    end
    nothing
end

function recompute_tile!(out,q,v,ratio,::Val{2},::VariableCarrier,tile)
    ny = size(q,2)
    ilo, ihi, jlo, jhi = tile
    for j in jlo:jhi
        jm = j == 1 ? ny : j-1
        @inbounds @simd for i in ilo:ihi
            out[i,j] = q[i,j] - ratio*(variable_yface(q,v,i,j,ratio,ny)-
                                       variable_yface(q,v,i,jm,ratio,ny))
        end
    end
    nothing
end

function flux_tile!(f,q,u,ratio,::Val{1},::VariableCarrier,tile)
    nx = size(q,1)
    ilo, ihi, jlo, jhi = tile
    for j in jlo:jhi
        @inbounds @simd for i in ilo:ihi
            f[i,j] = variable_xface(q,u,i,j,ratio,nx)
        end
    end
    nothing
end

function flux_tile!(f,q,v,ratio,::Val{2},::VariableCarrier,tile)
    ny = size(q,2)
    ilo, ihi, jlo, jhi = tile
    for j in jlo:jhi
        @inbounds @simd for i in ilo:ihi
            f[i,j] = variable_yface(q,v,i,j,ratio,ny)
        end
    end
    nothing
end

function gather_tile!(out, q, f, ratio, ::Val{1}, tile)
    nx = size(q,1)
    ilo, ihi, jlo, jhi = tile
    for j in jlo:jhi
        if ilo == 1
            @inbounds out[1,j] = q[1,j] - ratio*(f[1,j]-f[nx,j])
        end
        @inbounds @simd for i in max(2,ilo):ihi
            out[i,j] = q[i,j] - ratio*(f[i,j]-f[i-1,j])
        end
    end
    nothing
end

function gather_tile!(out, q, f, ratio, ::Val{2}, tile)
    ny = size(q,2)
    ilo, ihi, jlo, jhi = tile
    for j in jlo:jhi
        jm = j == 1 ? ny : j-1
        @inbounds @simd for i in ilo:ihi
            out[i,j] = q[i,j] - ratio*(f[i,j]-f[i,jm])
        end
    end
    nothing
end

function sweep!(out, q, velocity, ratio, direction, w::Workspace{B,Nothing,C}, pass) where {B,C}
    run_tiles!(tile -> recompute_tile!(out,q,velocity,ratio,direction,w.carrier,tile), w, pass)
    nothing
end

function sweep!(out, q, velocity, ratio, direction, w::Workspace{B,Matrix{Float64},C}, pass) where {B,C}
    run_tiles!(tile -> flux_tile!(w.flux,q,velocity,ratio,direction,w.carrier,tile), w, 2pass-1)
    # foreach_tile! synchronizes completely before any gather reads face fluxes.
    run_tiles!(tile -> gather_tile!(out,q,w.flux,ratio,direction,tile), w, 2pass)
    nothing
end

function validate_step(s::State, w::Workspace, dt)
    isfinite(dt) && dt >= 0 || throw(ArgumentError("timestep must be finite and nonnegative"))
    dims = size(s.q)
    dims == size(w.first) == size(w.second) == size(s.u) == size(s.v) ||
        throw(DimensionMismatch("state and workspace shapes must agree"))
    # Mutable public arrays must not be retargeted to input/output aliases.
    s.q === w.first || s.q === w.second || w.first === w.second ||
        s.q === s.u || s.q === s.v || w.first === s.u || w.first === s.v ||
        w.second === s.u || w.second === s.v ?
        throw(ArgumentError("input, output, and carrier arrays must be distinct")) : nothing
    if w.flux !== nothing
        size(w.flux) == dims || throw(DimensionMismatch("flux shape must agree"))
        any(a -> a === w.flux, (s.q,w.first,w.second,s.u,s.v)) &&
            throw(ArgumentError("flux buffer must be distinct"))
    end
    # Necessary constant-coefficient LW stability bound for each Strang substep.
    if w.carrier isa ConstantCarrier
        s.constant_carrier || throw(ArgumentError("use carrier=:variable for spatially varying speeds"))
        abs(s.u_speed)*dt*dims[1]*0.5 <= 1 && abs(s.v_speed)*dt*dims[2] <= 1 ||
            throw(ArgumentError("a directional Courant number exceeds one"))
    else
        # This is a conservative test fixture restriction, not a general
        # variable-coefficient stability theorem; smooth-flow tests are needed.
        maximum(abs,s.u)*dt*dims[1]*0.5 <= 0.4 &&
            maximum(abs,s.v)*dt*dims[2] <= 0.4 ||
            throw(ArgumentError("variable-carrier starting CFL 0.4 exceeded"))
    end
    nothing
end

"Advance one conservative Strang-composed constant-carrier LW timestep."
function step!(s::State, w::Workspace, dt::Real)
    delta = Float64(dt)
    validate_step(s,w,delta)
    nx, ny = size(s.q)
    sweep!(w.first,s.q,s.u,delta*nx*0.5,Val(1),w,1)
    sweep!(w.second,w.first,s.v,delta*ny,Val(2),w,2)
    sweep!(s.q,w.second,s.u,delta*nx*0.5,Val(1),w,3)
    s.t += delta
    s
end

"Advance a prescribed fixed-timestep trajectory; no diagnostic/output work is hidden."
function simulate!(s::State, w::Workspace, dt::Real, nsteps::Integer)
    nsteps >= 0 || throw(ArgumentError("step count must be nonnegative"))
    for _ in 1:nsteps
        step!(s,w,dt)
    end
    s
end

end
