using LinearAlgebra: I, eigvals
using Printf: @printf
using TwoPhaseAMR

function interface_mesh(block_size)
    return BlockTreeMesh(
        block_size=(block_size, block_size), max_level=2,
        refine=(key, origin, extent) ->
            key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0),
    )
end

function interface_error(block_size, interface_scheme; axis=1)
    mesh = interface_mesh(block_size)
    a, b = 0.3, -0.2
    velocity = FaceVelocity(
        mesh, (x, y, time, component) -> component == 1 ? a : b)
    field = axis == 1 ?
            FaceField{1}(mesh, (x, y) -> sinpi(2x) * cospi(2y)) :
            FaceField{2}(mesh, (x, y) -> sinpi(2x) * cospi(2y))
    rate = axis == 1 ? FaceField{1}(mesh) : FaceField{2}(mesh)
    workspace = StaggeredAdvectionWorkspace(field)
    dt = 0.05 / block_size
    convection!(rate, field, velocity, dt, workspace;
                scheme=LW(), amr_interface_scheme=interface_scheme)

    interface = falses(length(field.interior))
    for dual_face in workspace.schedule.faces
        dual_face.refinement_interface || continue
        interface[dual_face.minus] = true
        interface[dual_face.plus] = true
    end

    global_sum = interface_sum = regular_sum = 0.0
    interface_volume = regular_volume = 0.0
    local_index = 0
    for face in mesh.faces
        face.axis == axis || continue
        local_index += 1
        x, y = face.center
        exact = 2pi * (a * cospi(2x) * cospi(2y) -
                       b * sinpi(2x) * sinpi(2y))
        volume = TwoPhaseAMR._dual_volumes(workspace.schedule, local_index)
        contribution = volume * (rate.interior[local_index] - exact)^2
        global_sum += contribution
        if interface[local_index]
            interface_sum += contribution
            interface_volume += volume
        else
            regular_sum += contribution
            regular_volume += volume
        end
    end
    total_volume = interface_volume + regular_volume
    return (
        global_l2=sqrt(global_sum / total_volume),
        interface_l2=sqrt(interface_sum / interface_volume),
        regular_l2=sqrt(regular_sum / regular_volume),
        interface_volume_fraction=interface_volume / total_volume,
    )
end

function amplification_radius(mesh, ::Val{Axis}, interface_scheme;
                              cfl=0.8, sweep_forward=true) where {Axis}
    velocity = FaceVelocity(
        mesh, (x, y, time, component) -> component == 1 ? 0.3 : -0.2)
    basis = FaceField{Axis}(mesh)
    workspace = StaggeredAdvectionWorkspace(basis)
    dt = stable_timestep(velocity, workspace; cfl)
    n = length(basis.interior)
    update = zeros(n, n)
    for source in 1:n
        workspace.sweep_forward[] = sweep_forward
        fill!(basis.interior, 0)
        basis.interior[source] = 1
        advect!(basis, velocity, dt, workspace;
                scheme=LW(), amr_interface_scheme=interface_scheme)
        update[:, source] .= basis.interior
    end
    return maximum(abs, eigvals(update))
end

println("Mixed-level manufactured convection error")
for (name, interface_scheme) in
        (("LW / interface upwind", Upwind()),
         ("LW / interface Fromm", Fromm()),
         ("LW / interface minmod", Minmod()),
         ("LW / interface MC", MC()),
         ("LW / interface van Leer", VanLeer()),
         ("LW / interface superbee", Superbee()),
         ("LW throughout", LW()))
    results = interface_error.((4, 8, 16), Ref(interface_scheme))
    errors = getproperty.(results, :global_l2)
    rates = log2.(errors[1:2] ./ errors[2:3])
    println(name)
    for (block_size, result) in zip((4, 8, 16), results)
        @printf("  n=%2d global=%9.3e interface-rms=%9.3e regular-rms=%9.3e interface-volume=%7.3e\n",
                block_size, result.global_l2, result.interface_l2,
                result.regular_l2, result.interface_volume_fraction)
    end
    @printf("  global rates: %.3f, %.3f\n", rates...)
end

function interface_extremum_probe(::Val{Axis}, interface_scheme) where {Axis}
    mesh = interface_mesh(4)
    velocity = FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ? 0.3 : -0.2)
    field = FaceField{Axis}(mesh)
    rate = FaceField{Axis}(mesh)
    workspace = StaggeredAdvectionWorkspace(field)
    interface_incidence = zeros(Int, length(field.interior))
    for face in workspace.schedule.faces
        face.refinement_interface || continue
        interface_incidence[face.minus] += 1
        interface_incidence[face.plus] += 1
    end
    source = argmax(interface_incidence)
    field.interior[source] = 1
    convection!(rate, field, velocity, 0.0, workspace;
                scheme=LW(), amr_interface_scheme=interface_scheme)
    zero_nodes = findall(iszero, field.interior)
    return maximum(rate.interior[zero_nodes])
end

println("\nFrozen interface-extremum probe (positive means immediate undershoot)")
for (name, interface_scheme) in
        (("LW", LW()), ("Fromm", Fromm()), ("minmod", Minmod()),
         ("MC", MC()), ("van Leer", VanLeer()), ("superbee", Superbee()))
    violations = (interface_extremum_probe(Val(1), interface_scheme),
                  interface_extremum_probe(Val(2), interface_scheme))
    @printf("  %-9s x=%9.3e y=%9.3e\n", name, violations...)
end

println("\nOne-step spectral radii at CFL 0.8")
adaptive = interface_mesh(4)
uniform = BlockTreeMesh(block_size=(16, 16))
for axis in (1, 2)
    for sweep_forward in (true, false)
        order = sweep_forward ? "x/y" : "y/x"
        uniform_lw = amplification_radius(
            uniform, Val(axis), LW(); sweep_forward)
        adaptive_upwind = amplification_radius(
            adaptive, Val(axis), Upwind(); sweep_forward)
        adaptive_lw = amplification_radius(
            adaptive, Val(axis), LW(); sweep_forward)
        @printf("  component %d, %s: uniform LW %.15f, AMR hybrid %.15f, AMR LW %.15f\n",
                axis, order, uniform_lw, adaptive_upwind, adaptive_lw)
    end
end
