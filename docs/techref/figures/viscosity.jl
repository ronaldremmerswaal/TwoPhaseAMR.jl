# Chapter 9 (viscous stress and implicit solve): figure data and spot checks.
# Included by generate.jl after forces.jl, whose fixtures it reuses
# (force_mesh, force_drop, force_drop_fixture, FORCE_DENSITY) together with the
# helpers of vof.jl and momentum.jl. The functions marked "independent"
# rebuild the normal-stress rows, the stress weights and the staggered
# Laplacian of a uniform mesh without the package's operator code; the
# remaining checks evaluate exact algebraic identities (affine consistency,
# the implicit-midpoint energy and momentum balances, the common two-velocity
# increment) on the code's matrices and solves.

using SparseArrays: SparseMatrixCSC, rowvals, nonzeros, nzrange
using LinearAlgebra: dot

const VISC_PHASES = (0.1, 0.002)        # (liquid, gas) dynamic viscosity
const VISC_DT = 0.05

# The drop of Chapter 8 on the walled composite or cut mesh, or on a periodic
# composite mesh (no wall rows: momentum is conserved).
function viscosity_fixture(kind::Symbol)
    kind in (:amr, :cut) && return force_drop_fixture(kind)
    mesh = BlockTreeMesh(;
        base_mesh = (16, 16), block_size = (4, 4), extent = (FORCE_BOX, FORCE_BOX),
        periodic = (true, true), max_level = 1,
        refine = (key, origin, extent) -> key.level == 0 && key.i >= 2,
    )
    return InterfaceField(mesh, force_drop; lipschitz = 1)
end

variable_operator(field; boundary = NoSlip()) =
    StaggeredDiffusionOperator(field; viscosity = VISC_PHASES, boundary)

# Smooth test velocity (zero on no walls; boundary values are not unknowns).
visc_velocity(mesh; shift = 0.0) = FaceVelocity(
    mesh, (x, y, time, axis) -> axis == 1 ?
        sinpi(2x / FORCE_BOX + shift) * cospi(2y / FORCE_BOX) + 0.3 :
        -cospi(2x / FORCE_BOX) * sinpi(2y / FORCE_BOX + shift) - 0.2
)

cell_number(mesh, ref) = AMR._cell_index(ref, mesh.block_size[1], mesh.block_size[2])

# Regions of the shear rows in the order of `_shear_rows!` on an uncut mesh:
# every nonempty intersection of an x-velocity and a y-velocity contact region
# (read from the code's contact partitions), omitted where both derivatives
# vanish. Physical boxes (x0, y0, x1, y1), possibly a periodic image.
function shear_row_regions(mesh; boundary = NoSlip())
    lattice = AMR._finest_lattice_size(mesh)
    first, second = map((1, 2)) do axis
        AMR._shear_partition!(
            AMR.ShearPartition(Float64), mesh, lattice, axis, boundary, Float64,
        )
    end
    unit = mesh.extent ./ (2 .* lattice)
    function overlap(a_low, a_high, b_low, b_high, axis)
        for image in (mesh.periodic[axis] ? (-1, 0, 1) : (0,))
            shift = image * 2lattice[axis]
            low, high = max(a_low, b_low + shift), min(a_high, b_high + shift)
            high > low && return (low, high)
        end
        return nothing
    end
    derivative(partition, index) =
        partition.stencil_offsets[index + 1] > partition.stencil_offsets[index]
    regions = NTuple{4, Float64}[]
    for (i, a) in pairs(first.contacts), (j, b) in pairs(second.contacts)
        (derivative(first, i) || derivative(second, j)) || continue
        x = overlap(a.k_low, a.k_high, b.t_low, b.t_high, 1)
        y = overlap(a.t_low, a.t_high, b.k_low, b.k_high, 2)
        (isnothing(x) || isnothing(y)) && continue
        push!(
            regions, (
                mesh.origin[1] + x[1] * unit[1], mesh.origin[2] + y[1] * unit[2],
                mesh.origin[1] + x[2] * unit[1], mesh.origin[2] + y[2] * unit[2],
            ),
        )
    end
    return regions
end

# Area of a box inside a cell box, over the periodic images.
function box_overlap(mesh, box, cell)
    total = 0.0
    for sx in (mesh.periodic[1] ? (-1, 0, 1) : (0,)), sy in (mesh.periodic[2] ? (-1, 0, 1) : (0,))
        dx, dy = sx * mesh.extent[1], sy * mesh.extent[2]
        width = min(box[3], cell[3] + dx) - max(box[1], cell[1] + dx)
        height = min(box[4], cell[4] + dy) - max(box[2], cell[2] + dy)
        width > 0 && height > 0 && (total += width * height)
    end
    return total
end

# --- Independent normal rows and stress weights (Section "stress") ----------------

function stress_rows_check(field)
    mesh = field.mesh
    op = variable_operator(field)
    E = op.strain
    n = length(all_cells(mesh))
    geometry = field.geometry
    open(ref) = isnothing(geometry) ? 1.0 : AMR._get(geometry.open_fraction, ref)
    # Normal rows: (E u)_{c,k} = sum_f o_{c,f} a_f |f| u_f / |c^o| over the
    # interior faces of axis k of c.
    expected = zeros(2n, length(mesh.faces))
    for (index, face) in pairs(mesh.faces)
        a = isnothing(geometry) ? 1.0 : geometry.aperture[index]
        for (ref, o) in ((face.minus, 1.0), (face.plus, -1.0))
            alpha = open(ref)
            alpha > 0 || continue
            expected[2 * (cell_number(mesh, ref) - 1) + face.axis, index] +=
                o * a * face.area / (alpha * cell_volume(mesh, ref))
        end
    end
    normal_difference = maximum(abs, Matrix(E[1:(2n), :]) .- expected)
    # Weights: 2|c^o| mu_c with the harmonic cell mixture and, on an uncut
    # mesh, per shear row its area times the area-weighted harmonic mean of
    # the cells it covers.
    mu = Dict(
        ref => (c = cond(field, ref); inv(c / VISC_PHASES[1] + (1 - c) / VISC_PHASES[2]))
            for ref in all_cells(mesh)
    )
    weights = Float64[]
    for ref in all_cells(mesh), axis in 1:2
        alpha = open(ref)
        push!(weights, alpha > 0 ? 2 * alpha * cell_volume(mesh, ref) * mu[ref] : 0.0)
    end
    shear_rows = 0
    if isnothing(geometry)
        for box in shear_row_regions(mesh)
            parts = [(ref, box_overlap(mesh, box, cell_box_of(mesh, ref))) for ref in all_cells(mesh)]
            filter!(part -> part[2] > 0, parts)
            area = sum(last, parts)
            push!(weights, area^2 / sum(part[2] / mu[part[1]] for part in parts))
            shear_rows += 1
        end
        wall_rows = length(op.stress_weights) - length(weights)
    else
        wall_rows = length(
            AMR._embedded_variable_wall_stresses(
                geometry; include_cut_wall = true, dirichlet_faces = op.dirichlet_faces,
            ),
        ) + length(op.dirichlet_faces)
    end
    count = length(weights)
    # Covered cells have zero normal-row measure; compare the open rows.
    covered = [i for i in 1:count if weights[i] == 0]
    @assert all(i -> op.stress_weights[i] == 0, covered)
    weight_difference = maximum(
        abs(op.stress_weights[i] - weights[i]) / weights[i]
            for i in 1:count if weights[i] > 0
    )
    return (;
        normal_difference, weight_difference, normal_rows = 2n, shear_rows, wall_rows,
    )
end

# Affine consistency: for the six affine basis velocities every interior
# stress row (not touching a wall) returns the exact strain rate.
function affine_strain_check(field)
    mesh = field.mesh
    op = variable_operator(field)
    E = op.strain
    n = length(all_cells(mesh))
    hmax = maximum(maximum(block.spacing) for block in mesh.blocks)
    # Rows: normal rows of cells without a boundary face and farther than
    # 4 h_max from the domain edges (periodic seams: the affine field jumps
    # there); shear rows whose region is farther than 4 h_max from the edges.
    boundary_cells = Set(face.cell for face in mesh.boundary_faces)
    edge_distance(point) = minimum(min(point[k] - mesh.origin[k], mesh.origin[k] + mesh.extent[k] - point[k]) for k in 1:2)
    rows = Int[]
    exact_index = Tuple{Symbol, Int}[]
    for ref in all_cells(mesh), axis in 1:2
        ref in boundary_cells && continue
        edge_distance(cellcenter(mesh, ref.block, ref.i, ref.j)) > 4hmax || continue
        push!(rows, 2 * (cell_number(mesh, ref) - 1) + axis)
        push!(exact_index, (:normal, axis))
    end
    for (offset, box) in enumerate(shear_row_regions(mesh))
        min(edge_distance((box[1], box[2])), edge_distance((box[3], box[4]))) > 4hmax || continue
        push!(rows, 2n + offset)
        push!(exact_index, (:shear, 0))
    end
    worst = 0.0
    for component in 1:2, gradient in 1:2
        velocity = FaceVelocity(mesh, (x, y, time, axis) -> axis == component ? (gradient == 1 ? x : y) : 0.0)
        rates = E * velocity.interior
        for (row, (kind, axis)) in zip(rows, exact_index)
            exact = kind == :normal ? Float64(axis == component && gradient == component) :
                Float64(component != gradient)
            worst = max(worst, abs(rates[row] - exact))
        end
    end
    for component in 1:2   # constants: no strain
        velocity = FaceVelocity(mesh, (x, y, time, axis) -> axis == component ? 1.0 : 0.0)
        rates = E * velocity.interior
        worst = max(worst, maximum(abs, rates[rows]))
    end
    return (; worst, rows = length(rows))
end

# --- Implicit midpoint: residual, energy and momentum (Section "time") -------------

function midpoint_check(kind::Symbol)
    field = viscosity_fixture(kind)
    mesh = field.mesh
    op = variable_operator(field)
    AMR._update_dynamic_viscosity!(op, field, VISC_PHASES)
    density = mixture_density(field, FORCE_DENSITY)
    solver = AMR.VariableStaggeredDiffusionSolver(op, density, VISC_DT)
    source = visc_velocity(mesh)
    result = FaceVelocity(mesh)
    implicit_midpoint_diffusion!(result, source, op, VISC_DT, solver; density, tolerance = 1.0e-13, max_cycles = 2000)
    M = density .* op.volumes
    S = op.stiffness
    us, uss = source.interior, result.interior
    residual = M .* (uss .- us) .+ (VISC_DT / 2) .* (S * (uss .+ us))
    strong = maximum(abs, residual ./ M) / maximum(abs, us)
    sum_velocity = us .+ uss
    dissipation = VISC_DT / 4 * dot(sum_velocity, S * sum_velocity)
    energy_change = (dot(uss, M .* uss) - dot(us, M .* us)) / 2
    energy_defect = abs(energy_change + dissipation) / dissipation
    momentum = 0.0
    constant_stress = 0.0
    if all(mesh.periodic) && isnothing(field.geometry)
        for k in 1:2
            ones_k = [face.axis == k ? 1.0 : 0.0 for face in mesh.faces]
            before = dot(ones_k, M .* us)
            after = dot(ones_k, M .* uss)
            momentum = max(momentum, abs(after - before) / dot(ones_k, M .* abs.(us)))
            constant_stress = max(constant_stress, maximum(abs, S * ones_k) / maximum(abs, nonzeros(S)))
        end
    end
    return (;
        strong, energy_defect, dissipation_ratio = dissipation / (dot(us, M .* us) / 2),
        momentum, constant_stress,
    )
end

# Two velocities: mixture velocity, common increment and unchanged jump.
function two_velocity_viscosity_check()
    field = viscosity_fixture(:amr)
    mesh = field.mesh
    physics = TwoPhase(field; density = FORCE_DENSITY, dynamic_viscosity = VISC_PHASES)
    model = AMR.TwoPhaseModel(physics)
    op = variable_operator(field)
    AMR._update_dynamic_viscosity!(op, field, VISC_PHASES)
    liquid = visc_velocity(mesh)
    gas = visc_velocity(mesh; shift = 0.37)
    velocity = (liquid, gas)
    mixture = FaceVelocity(mesh)
    density = zeros(length(mesh.faces))
    AMR._mass_weighted_one_velocity!(mixture, density, velocity, model)
    worst_mixture = 0.0
    for (index, face) in pairs(mesh.faces)
        c = staggered_fraction_independent(field, face)
        m_l, m_g = FORCE_DENSITY[1] * c, FORCE_DENSITY[2] * (1 - c)
        expected = (m_l * liquid.interior[index] + m_g * gas.interior[index]) / (m_l + m_g)
        worst_mixture = max(worst_mixture, abs(mixture.interior[index] - expected))
    end
    solver = AMR.VariableStaggeredDiffusionSolver(op, density, VISC_DT)
    diffused = FaceVelocity(mesh)
    implicit_midpoint_diffusion!(diffused, mixture, op, VISC_DT, solver; density, tolerance = 1.0e-13, max_cycles = 2000)
    updated = (FaceVelocity(mesh), FaceVelocity(mesh))
    AMR._apply_two_velocity_diffusion_update!(updated, velocity, diffused, mixture)
    jump_before = gas.interior .- liquid.interior
    jump_after = updated[2].interior .- updated[1].interior
    jump = maximum(abs, jump_after .- jump_before) / maximum(abs, jump_before)
    recombined = FaceVelocity(mesh)
    AMR._mass_weighted_one_velocity!(recombined, zeros(length(mesh.faces)), updated, model)
    recombination = maximum(abs, recombined.interior .- diffused.interior) / maximum(abs, diffused.interior)
    return (; worst_mixture, jump, recombination)
end

# --- Single-phase staggered Laplacian (Section "single phase") -------------------------

function face_lookup(mesh, axis)
    table = Dict{NTuple{2, Int}, Int}()
    for (index, face) in pairs(mesh.faces)
        face.axis == axis || continue
        table[(round(Int, 4face.center[1]), round(Int, 4face.center[2]))] = index
    end
    return table
end

# Uniform periodic mesh: the operator is the five-point Laplacian of each
# component (independent stencil by face lookup).
function uniform_laplacian_check()
    mesh = BlockTreeMesh(base_mesh = (16, 16), block_size = (4, 4), extent = (8.0, 8.0), periodic = (true, true))
    op = StaggeredDiffusionOperator(mesh; viscosity = 1.0)
    velocity = visc_velocity(mesh)
    rate = FaceVelocity(mesh)
    diffusion!(rate, velocity, op)
    h = 0.5
    worst = 0.0
    for axis in 1:2
        table = face_lookup(mesh, axis)
        wrapq(i, k) = mod(i, round(Int, 4mesh.extent[k]))
        for (key, index) in table
            neighbours = (
                (wrapq(key[1] + 2, 1), key[2]), (wrapq(key[1] - 2, 1), key[2]),
                (key[1], wrapq(key[2] + 2, 2)), (key[1], wrapq(key[2] - 2, 2)),
            )
            value = (sum(velocity.interior[table[n]] for n in neighbours) - 4velocity.interior[index]) / h^2
            worst = max(worst, abs(rate.interior[index] - value))
        end
    end
    return worst / maximum(abs, rate.interior)
end

# Composite walled mesh: check the actual volume-weighted operator and its
# symmetric positive semidefinite stiffness, including the AMR contacts.
function symmetric_laplacian_check()
    mesh = force_mesh(:amr)
    op = StaggeredDiffusionOperator(mesh; viscosity = 1.0)
    relation, asymmetry, minimum_eigenvalue = 0.0, 0.0, Inf
    for component in (op.x, op.y)
        stiffness = component.stiffness
        integrated = spdiagm(0 => component.volumes) * component.laplacian
        scale = max(1.0, maximum(abs, nonzeros(stiffness); init = 0.0))
        relation = max(
            relation, maximum(abs, nonzeros(integrated + stiffness); init = 0.0) / scale,
        )
        asymmetry = max(
            asymmetry,
            maximum(abs, nonzeros(integrated - transpose(integrated)); init = 0.0) / scale,
        )
        minimum_eigenvalue = min(
            minimum_eigenvalue, eigmin(Symmetric(Matrix(stiffness))) / scale,
        )
    end
    return (; relation, asymmetry, minimum_eigenvalue)
end

# Single-phase implicit midpoint: residual of
# (I - dt nu L / 2) u = u_c + dt nu L u_old / 2 (stationary no-slip walls).
function single_phase_midpoint_check()
    mesh = force_mesh(:amr)
    nu = 0.01
    op = StaggeredDiffusionOperator(mesh; viscosity = nu)
    dt = 0.05
    solver = StaggeredEllipticSolver(StaggeredEllipticOperator(op; diffusion = dt * nu / 2))
    convected = visc_velocity(mesh)
    old = visc_velocity(mesh; shift = 0.11)
    result = FaceVelocity(mesh)
    work = FaceVelocity(mesh)
    implicit_midpoint_diffusion!(result, convected, old, op, dt, solver, work; tolerance = 1.0e-13, max_cycles = 500)
    worst = 0.0
    for component in (op.x, op.y)
        g = component.local_to_global
        L = component.laplacian
        lhs = result.interior[g] .- (dt * nu / 2) .* (L * result.interior[g])
        rhs = convected.interior[g] .+ (dt * nu / 2) .* (L * old.interior[g])
        worst = max(worst, maximum(abs, lhs .- rhs))
    end
    return worst / maximum(abs, convected.interior)
end

# Suspected defect: a moving NoSlip(f) also moves the embedded solid walls.
function moving_wall_check()
    mesh = BlockTreeMesh(base_mesh = (16, 16), block_size = (4, 4), extent = (8.0, 8.0), periodic = (false, false))
    geometry = GeometryField(mesh, (x, y) -> hypot(x - 4.0, y - 3.2) - 1.6; lipschitz = 1)
    lid = (x, y, time, axis) -> axis == 1 ? 1.0 : 0.0
    near = [i for (i, face) in pairs(mesh.faces) if hypot(face.center[1] - 4.0, face.center[2] - 3.2) < 2.4]
    rates = map((NoSlip(lid), VelocityBoundaryConditions(top = NoSlip(lid)))) do boundary
        op = StaggeredDiffusionOperator(geometry; viscosity = 1.0, boundary)
        rate = FaceVelocity(mesh)
        diffusion!(rate, FaceVelocity(mesh), op)
        maximum(abs, rate.interior[near])
    end
    return (; bare = rates[1], sides = rates[2])
end

# --- Figure data: rows of E near a level transition and a wall --------------------------

const VISC_WINDOW = (3.0, 5.0, 0.0, 1.25)   # x0, x1, y0, y1

#   \fxwcell{x0}{y0}{x1}{y1}{level}   \fxwhanging{x}{y}
#   \fxwface{axis}{x}{y}{length}{panel role: 0 plain, 1 target}
#   \fxwrow{kind}{x}{y}{coefficient}  kind 0 normal row (cell centre), 1 shear row (region centre)
#   \fxwstencil{axis}{x}{y}{coefficient}  shear stencil at the hanging vertex
#   (coefficients rounded to three significant digits for the labels)
coefficient_label(value) = replace(@sprintf("%.3g", value), "e" => "\\mathrm{e}")
function viscosity_data()
    mesh = force_mesh(:amr)
    field = InterfaceField(mesh, (x, y) -> 1.0)    # all gas: geometry of E only
    op = StaggeredDiffusionOperator(field; viscosity = (1.0, 1.0))
    E = op.strain
    n = length(all_cells(mesh))
    io = IOBuffer()
    inside(x, y) = VISC_WINDOW[1] - 1.0e-9 <= x <= VISC_WINDOW[2] + 1.0e-9 &&
        VISC_WINDOW[3] - 1.0e-9 <= y <= VISC_WINDOW[4] + 1.0e-9
    for ref in all_cells(mesh)
        box = cell_box_of(mesh, ref)
        (box[3] > VISC_WINDOW[1] && box[1] < VISC_WINDOW[2] && box[4] > VISC_WINDOW[3] && box[2] < VISC_WINDOW[4]) || continue
        println(io, "\\fxwcell{$(fmt(box[1]))}{$(fmt(box[2]))}{$(fmt(box[3]))}{$(fmt(box[4]))}{$(cell_level(mesh, ref))}")
    end
    target = findfirst(
        face -> face.axis == 1 && abs(face.center[1] - 4.0) < 1.0e-12 &&
            abs(face.center[2] - 0.125) < 1.0e-12, mesh.faces
    )
    @assert !isnothing(target)
    for (index, face) in pairs(mesh.faces)
        inside(face.center...) || continue
        println(io, "\\fxwface{$(face.axis)}{$(fmt(face.center[1]))}{$(fmt(face.center[2]))}{$(fmt(face.area))}{$(index == target ? 1 : 0)}")
    end
    # Rows containing the target unknown: cell centres of normal rows,
    # region centres of shear rows.
    regions = shear_row_regions(mesh)
    @assert size(E, 1) == 2n + length(regions)
    region_centre(row) = let box = regions[row - 2n]
        ((box[1] + box[3]) / 2, (box[2] + box[4]) / 2)
    end
    rows_of_target = Tuple{Int, Float64}[]
    for row in 1:size(E, 1)
        value = E[row, target]
        iszero(value) || push!(rows_of_target, (row, value))
    end
    info = Dict(:normal => 0, :shear => 0)
    for (row, value) in rows_of_target
        if row <= 2n
            cell = div(row - 1, 2) + 1
            block = div(cell - 1, prod(mesh.block_size)) + 1
            local_index = rem(cell - 1, prod(mesh.block_size))
            ref = CellRef(block, rem(local_index, mesh.block_size[1]) + 1, div(local_index, mesh.block_size[1]) + 1)
            x = cellcenter(mesh, ref.block, ref.i, ref.j)
            println(io, "\\fxwrow{0}{$(fmt(x[1]))}{$(fmt(x[2]))}{$(coefficient_label(value))}")
            info[:normal] += 1
        else
            p = region_centre(row)
            println(io, "\\fxwrow{1}{$(fmt(p[1]))}{$(fmt(p[2]))}{$(coefficient_label(value))}")
            info[:shear] += 1
        end
    end
    # The shear rows whose regions meet at the hanging vertex (4, 1/4): the
    # x-velocity contact through the vertex meets the two halves of the
    # anchored y-velocity group, so both rows have the same stencil.
    hanging_rows = [
        2n + index for (index, box) in pairs(regions) if
            box[1] < 4.0 < box[3] && box[2] <= 0.25 <= box[4]
    ]
    @assert length(hanging_rows) == 2 && E[hanging_rows[1], :] == E[hanging_rows[2], :]
    hanging_row = first(hanging_rows)
    println(io, "\\fxwhanging{4}{0.25}")
    stencil_faces = 0
    for (index, face) in pairs(mesh.faces)
        value = E[hanging_row, index]
        iszero(value) && continue
        stencil_faces += 1
        println(io, "\\fxwstencil{$(face.axis)}{$(fmt(face.center[1]))}{$(fmt(face.center[2]))}{$(coefficient_label(value))}")
    end
    info[:stencil] = stencil_faces
    return String(take!(io)), info
end

# --- Outputs -------------------------------------------------------------------

function viscosity_outputs(write_output)
    data, figure_info = viscosity_data()
    write_output("viscosity.tex", data)

    rows = Dict(kind => stress_rows_check(viscosity_fixture(kind)) for kind in (:amr, :cut, :periodic))
    affine = Dict(kind => affine_strain_check(viscosity_fixture(kind)) for kind in (:amr, :periodic))
    midpoint = Dict(kind => midpoint_check(kind) for kind in (:amr, :cut, :periodic))
    two = two_velocity_viscosity_check()
    uniform = uniform_laplacian_check()
    symmetric = symmetric_laplacian_check()
    single = single_phase_midpoint_check()
    wall = moving_wall_check()

    println("viscosity stress rows:       ", rows)
    println("viscosity affine strain:     ", affine)
    println("viscosity midpoint:          ", midpoint)
    println("viscosity two velocities:    ", two)
    println("viscosity single phase:      ", (; uniform, symmetric, single))
    println("viscosity moving wall:       ", wall)
    println("viscosity figure:            ", figure_info)

    normal = maximum(r.normal_difference for r in values(rows))
    weight = maximum(r.weight_difference for r in values(rows))
    affine_worst = maximum(a.worst for a in values(affine))
    strong = maximum(m.strong for m in values(midpoint))
    energy = maximum(m.energy_defect for m in values(midpoint))
    @assert normal <= 1.0e-12 && weight <= 1.0e-13 && affine_worst <= 1.0e-12
    @assert strong <= 1.0e-11 && energy <= 1.0e-9
    @assert midpoint[:periodic].momentum <= 1.0e-12 && midpoint[:periodic].constant_stress <= 1.0e-12
    @assert two.worst_mixture <= 1.0e-13 && two.jump <= 1.0e-14 && two.recombination <= 1.0e-13
    @assert uniform <= 1.0e-13 && single <= 1.0e-11
    @assert symmetric.relation <= 1.0e-12 && symmetric.asymmetry <= 1.0e-12
    @assert symmetric.minimum_eigenvalue >= -1.0e-12
    @assert wall.bare == 0 && wall.sides == 0

    return Any[
        ("chkViscNormalDiff", sci(normal)),
        ("chkViscWeightDiff", sci(weight)),
        ("chkViscShearRows", rows[:amr].shear_rows),
        ("chkViscWallRows", rows[:cut].wall_rows),
        ("chkViscAffine", sci(affine_worst)),
        ("chkViscAffineRows", affine[:amr].rows),
        ("chkViscStrong", sci(strong)),
        ("chkViscEnergy", sci(energy)),
        ("chkViscMomentum", sci(midpoint[:periodic].momentum)),
        ("chkViscConstant", sci(midpoint[:periodic].constant_stress)),
        ("chkViscMixture", sci(two.worst_mixture)),
        ("chkViscJump", sci(two.jump)),
        ("chkViscRecombine", sci(two.recombination)),
        ("chkViscUniform", sci(uniform)),
        ("chkViscRelation", sci(symmetric.relation)),
        ("chkViscWeightedAsym", sci(symmetric.asymmetry)),
        ("chkViscMinEig", sci(symmetric.minimum_eigenvalue)),
        ("chkViscSingleMidpoint", sci(single)),
        ("chkViscFigNormal", figure_info[:normal]),
        ("chkViscFigShear", figure_info[:shear]),
        ("chkViscFigStencil", figure_info[:stencil]),
    ]
end
