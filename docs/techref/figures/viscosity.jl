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
visc_velocity(mesh; shift = 0.0) = FaceVelocity(mesh, (x, y, time, axis) -> axis == 1 ?
    sinpi(2x / FORCE_BOX + shift) * cospi(2y / FORCE_BOX) + 0.3 :
    -cospi(2x / FORCE_BOX) * sinpi(2y / FORCE_BOX + shift) - 0.2)

cell_number(mesh, ref) = AMR._cell_index(ref, mesh.block_size[1], mesh.block_size[2])

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
    # Weights: 2|c^o| mu_c and, per corner vertex, |omega^v| exp(mean log mu).
    logmu = Dict(ref => (c = cond(field, ref); c * log(VISC_PHASES[1]) + (1 - c) * log(VISC_PHASES[2]))
        for ref in all_cells(mesh))
    weights = Float64[]
    for ref in all_cells(mesh), axis in 1:2
        alpha = open(ref)
        push!(weights, alpha > 0 ? 2 * alpha * cell_volume(mesh, ref) * exp(logmu[ref]) : 0.0)
    end
    lattice = mesh.base_mesh .* 2^mesh.max_level
    vertices = Dict{NTuple{2, Int}, Vector{Tuple{CellRef, Float64}}}()
    for ref in all_cells(mesh)
        x = cellcenter(mesh, ref.block, ref.i, ref.j)
        h = cell_spacing(mesh, ref)
        for sx in (-1, 1), sy in (-1, 1)
            key = ntuple(2) do k
                value = round(Int, (x[k] + (k == 1 ? sx : sy) * h[k] / 2 - mesh.origin[k]) / mesh.extent[k] * lattice[k])
                mesh.periodic[k] ? mod(value, lattice[k]) : value
            end
            push!(get!(vertices, key, Tuple{CellRef, Float64}[]), (ref, cell_volume(mesh, ref) / 4))
        end
    end
    for key in sort!(collect(keys(vertices)))
        cells = vertices[key]
        volume = sum(w * open(ref) for (ref, w) in cells)
        volume > 0 || continue
        # Covered cells (no open volume) do not enter the vertex mean.
        logsum = sum(w * open(ref) * logmu[ref] for (ref, w) in cells if open(ref) > 0)
        push!(weights, volume * exp(logsum / volume))
    end
    count = length(weights)
    # Covered cells have zero normal-row measure; compare the open rows.
    covered = [i for i in 1:count if weights[i] == 0]
    @assert all(i -> op.stress_weights[i] == 0, covered)
    weight_difference = maximum(abs(op.stress_weights[i] - weights[i]) / weights[i]
        for i in 1:count if weights[i] > 0)
    return (; normal_difference, weight_difference, normal_rows = 2n,
        shear_rows = count - 2n, wall_rows = length(op.stress_weights) - count)
end

# Affine consistency: for the six affine basis velocities every interior
# stress row (not touching a wall) returns the exact strain rate.
function affine_strain_check(field)
    mesh = field.mesh
    op = variable_operator(field)
    E = op.strain
    n = length(all_cells(mesh))
    lattice = mesh.base_mesh .* 2^mesh.max_level
    hmax = maximum(maximum(block.spacing) for block in mesh.blocks)
    # Rows: normal rows of cells without a boundary face and farther than
    # 4 h_max from the domain edges (periodic seams: the affine field jumps
    # there); shear rows of vertices farther than 4 h_max from the edges.
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
    vertex_key(x, h, sx, sy) = ntuple(2) do k
        value = round(Int, (x[k] + (k == 1 ? sx : sy) * h[k] / 2 - mesh.origin[k]) / mesh.extent[k] * lattice[k])
        mesh.periodic[k] ? mod(value, lattice[k]) : value
    end
    keys_sorted = sort!(unique([
        vertex_key(cellcenter(mesh, ref.block, ref.i, ref.j), cell_spacing(mesh, ref), sx, sy)
            for ref in all_cells(mesh) for sx in (-1, 1) for sy in (-1, 1)
    ]))
    for (offset, key) in enumerate(keys_sorted)
        point = ntuple(k -> mesh.origin[k] + key[k] * mesh.extent[k] / lattice[k], 2)
        distance = edge_distance(point)
        distance > 4hmax || continue
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
    return (; strong, energy_defect, dissipation_ratio = dissipation / (dot(us, M .* us) / 2),
        momentum, constant_stress)
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

# Composite walled mesh: rows of the corrected band reproduce the Laplacian of
# quadratics; count band rows and the asymmetry of the operator.
function corrected_laplacian_check()
    mesh = force_mesh(:amr)
    op = StaggeredDiffusionOperator(mesh; viscosity = 1.0)
    worst = 0.0
    band = 0
    asymmetric = 0
    for component in (op.x, op.y)
        L = component.laplacian
        base = -spdiagm(0 => inv.(component.volumes)) * component.stiffness
        points = [mesh.faces[g].center for g in component.local_to_global]
        rows = [r for r in 1:size(L, 1) if maximum(abs, L[r, :] .- base[r, :]; init = 0.0) > 1.0e-12]
        band += length(rows)
        for q in ((x, y) -> x^2, (x, y) -> x * y, (x, y) -> y^2, (x, y) -> x, (x, y) -> 1.0)
            values = [q(p...) for p in points]
            exact = q(2.0, 3.0) == 4.0 ? 2.0 : q(2.0, 3.0) == 9.0 ? 2.0 : 0.0
            laplacian = L * values
            worst = max(worst, maximum(abs, laplacian[rows] .- exact; init = 0.0))
        end
        scaled = spdiagm(0 => component.volumes) * L
        asymmetric += count(!iszero, abs.(scaled .- transpose(scaled)) .> 1.0e-12 * maximum(abs, nonzeros(scaled)))
    end
    return (; worst, band, asymmetric)
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
#   \fxwrow{kind}{x}{y}{coefficient}  kind 0 normal row (cell centre), 1 shear row (vertex)
#   \fxwstencil{axis}{x}{y}{coefficient}  shear stencil of the hanging vertex
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
    target = findfirst(face -> face.axis == 1 && abs(face.center[1] - 4.0) < 1.0e-12 &&
        abs(face.center[2] - 0.125) < 1.0e-12, mesh.faces)
    @assert !isnothing(target)
    for (index, face) in pairs(mesh.faces)
        inside(face.center...) || continue
        println(io, "\\fxwface{$(face.axis)}{$(fmt(face.center[1]))}{$(fmt(face.center[2]))}{$(fmt(face.area))}{$(index == target ? 1 : 0)}")
    end
    # Rows containing the target unknown: cell centres of normal rows,
    # vertices of shear rows (located through their stencils).
    lattice = mesh.base_mesh .* 2^mesh.max_level
    keys_sorted = sort!(unique([
        let x = cellcenter(mesh, ref.block, ref.i, ref.j), h = cell_spacing(mesh, ref)
            (round(Int, (x[1] + sx * h[1] / 2 - mesh.origin[1]) / mesh.extent[1] * lattice[1]),
                round(Int, (x[2] + sy * h[2] / 2 - mesh.origin[2]) / mesh.extent[2] * lattice[2]))
        end for ref in all_cells(mesh) for sx in (-1, 1) for sy in (-1, 1)
    ]))
    vertex_point(row) = let key = keys_sorted[row - 2n]
        ntuple(k -> mesh.origin[k] + key[k] * mesh.extent[k] / lattice[k], 2)
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
            p = vertex_point(row)
            println(io, "\\fxwrow{1}{$(fmt(p[1]))}{$(fmt(p[2]))}{$(coefficient_label(value))}")
            info[:shear] += 1
        end
    end
    # The shear row of the hanging vertex (4, 1/4): its whole stencil.
    hanging_key = (round(Int, 4.0 / mesh.extent[1] * lattice[1]), round(Int, 0.25 / mesh.extent[2] * lattice[2]))
    hanging_row = 2n + findfirst(==(hanging_key), keys_sorted)
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
    corrected = corrected_laplacian_check()
    single = single_phase_midpoint_check()
    wall = moving_wall_check()

    println("viscosity stress rows:       ", rows)
    println("viscosity affine strain:     ", affine)
    println("viscosity midpoint:          ", midpoint)
    println("viscosity two velocities:    ", two)
    println("viscosity single phase:      ", (; uniform, corrected, single))
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
    @assert uniform <= 1.0e-13 && corrected.worst <= 1.0e-9 && single <= 1.0e-11
    @assert wall.bare > 1 && wall.sides == 0

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
        ("chkViscBand", corrected.band),
        ("chkViscBandQuadratic", sci(corrected.worst)),
        ("chkViscAsymmetric", corrected.asymmetric),
        ("chkViscSingleMidpoint", sci(single)),
        ("chkViscLidBare", @sprintf("%.0f", wall.bare)),
        ("chkViscFigNormal", figure_info[:normal]),
        ("chkViscFigShear", figure_info[:shear]),
        ("chkViscFigStencil", figure_info[:stencil]),
    ]
end
