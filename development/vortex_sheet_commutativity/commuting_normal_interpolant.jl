module VortexSheetCommutativity

using LinearAlgebra
using Printf
using SparseArrays
using TwoPhaseAMR

const DEFAULT_CENTRE = (0.5, 0.5)
const DEFAULT_RADIUS = 0.27
const DEFAULT_DENSITY = (10.0, 1.0)

"""One fixed mesh/interface/pressure instance; `N` is deliberately absent."""
struct CommutativityProblem{T}
    resolution::Int
    mesh
    interface
    assembly
    density::NTuple{2, T}
    centre::NTuple{2, T}
    radius::T
    active_faces::Vector{Int}
    kinetic_diagonal::Vector{T}
    constant_samples::Matrix{T}
    pressure_jump_gradient::Vector{T}
    tangent::SparseMatrixCSC{T, Int}
    normal_baseline::SparseMatrixCSC{T, Int}
    target_tangent::Vector{T}
    row_cells::Vector{Int}
    row_points::Vector{NTuple{2, T}}
    row_normals::Vector{NTuple{2, T}}
    row_tangents::Vector{NTuple{2, T}}
    reconstruction_radius::T
    active_mass_fraction::T
end

"""A candidate normal interpolant and the geometry of its output rows."""
struct NormalCandidate{T}
    name::Symbol
    matrix::SparseMatrixCSC{T, Int}
    points::Vector{NTuple{2, T}}
    normals::Vector{NTuple{2, T}}
    construction_residual::T
    mean_support::T
    maximum_support::Int
end

"""Diagnostics from one candidate at one resolution."""
struct CandidateDiagnostics{T}
    resolution::Int
    name::Symbol
    rows::Int
    normal_constant_error::T
    normal_l2_error::T
    normal_linf_error::T
    pressure_l2_error::T
    pressure_linf_error::T
    combined_l2_error::T
    combined_linf_error::T
    multiplier_l2_error::T
    multiplier_linf_error::T
    construction_residual::T
    mean_support::T
    maximum_support::Int
end

"""
    phase_pressure(phase, x, y)

Smooth periodic one-sided pressure fields.  Edit these two functions to
change the manufactured pressure data without changing any operator code.
"""
function phase_pressure(phase, x, y)
    phase in (1, 2) || throw(ArgumentError("phase must be 1 or 2"))
    return 1.1sinpi(2x) + 0.35cospi(2y) + 0.2sinpi(2(x + y))
end

"""Analytic gradient of [`phase_pressure`](@ref)."""
function phase_pressure_gradient(phase, x, y)
    phase in (1, 2) || throw(ArgumentError("phase must be 1 or 2"))
    return (
        2.2pi * cospi(2x) + 0.4pi * cospi(2(x + y)),
        -0.7pi * sinpi(2y) + 0.4pi * cospi(2(x + y)),
    )
end

@inline function pressure_potential_gradient(density, x, y)
    liquid = phase_pressure_gradient(1, x, y)
    gas = phase_pressure_gradient(2, x, y)
    return (
        gas[1] / density[2] - liquid[1] / density[1],
        gas[2] / density[2] - liquid[2] / density[1],
    )
end

@inline function manufactured_jump(x, y)
    return (
        sinpi(2x) + 0.3cospi(2y) + 0.15sinpi(2(x + y)),
        cospi(2y) - 0.2sinpi(2x) + 0.1cospi(2(x - y)),
    )
end

@inline dot2(a, b) = a[1] * b[1] + a[2] * b[2]

@inline function periodic_delta(a, b)
    distance = abs(a - b)
    return min(distance, one(distance) - distance)
end

function circle_point_and_frame(point, centre, radius)
    dx = point[1] - centre[1]
    dy = point[2] - centre[2]
    distance = hypot(dx, dy)
    distance > 0 || error("the circle centre cannot be an interface row")
    normal = (dx / distance, dy / distance)
    tangent = (-normal[2], normal[1])
    interface_point = (
        centre[1] + radius * normal[1],
        centre[2] + radius * normal[2],
    )
    return interface_point, normal, tangent
end

function active_constant_samples(mesh, active_faces, ::Type{T}) where {T}
    result = zeros(T, length(active_faces), 2)
    for (local_face, global_face) in pairs(active_faces)
        axis = mesh.faces[global_face].axis
        result[local_face, axis] = one(T)
    end
    return result
end

function nearby_axis_faces(
        mesh, active_faces, point, axis, h, initial_radius,
    )
    for radius in initial_radius:one(initial_radius):initial_radius + 4
        candidates = Tuple{Int, Float64}[]
        limit = radius * h
        for (local_face, global_face) in pairs(active_faces)
            face = mesh.faces[global_face]
            face.axis == axis || continue
            dx = periodic_delta(face.center[1], point[1])
            dy = periodic_delta(face.center[2], point[2])
            max(dx, dy) <= limit || continue
            push!(candidates, (local_face, hypot(dx, dy)))
        end
        isempty(candidates) || return candidates
    end
    return Tuple{Int, Float64}[]
end

function reconstruction_weights(candidates, h)
    raw = [inv(distance^2 + (0.25h)^2) for (_, distance) in candidates]
    raw ./= sum(raw)
    return raw
end

function fixed_tangent_reconstruction(
        mesh, active_faces, cell_indices, points, normals, tangents;
        reconstruction_radius,
    )
    T = eltype(normals[1])
    h = mesh.blocks[1].spacing[1]
    tangent_rows = Int[]
    tangent_columns = Int[]
    tangent_values = T[]
    normal_rows = Int[]
    normal_columns = Int[]
    normal_values = T[]
    retained_cells = Int[]
    retained_points = NTuple{2, T}[]
    retained_normals = NTuple{2, T}[]
    retained_tangents = NTuple{2, T}[]
    maximum_radius = zero(T)
    for row in eachindex(cell_indices)
        candidates = ntuple(2) do axis
            nearby_axis_faces(
                mesh, active_faces, points[row], axis, h,
                reconstruction_radius,
            )
        end
        any(isempty, candidates) && continue
        new_row = length(retained_cells) + 1
        for axis in 1:2
            weights = reconstruction_weights(candidates[axis], h)
            for ((column, distance), weight) in zip(
                    candidates[axis], weights,
                )
                push!(tangent_rows, new_row)
                push!(tangent_columns, column)
                push!(tangent_values, tangents[row][axis] * weight)
                push!(normal_rows, new_row)
                push!(normal_columns, column)
                push!(normal_values, normals[row][axis] * weight)
                maximum_radius = max(maximum_radius, T(distance / h))
            end
        end
        push!(retained_cells, cell_indices[row])
        push!(retained_points, points[row])
        push!(retained_normals, normals[row])
        push!(retained_tangents, tangents[row])
    end
    number_of_rows = length(retained_cells)
    number_of_faces = length(active_faces)
    tangent = sparse(
        tangent_rows, tangent_columns, tangent_values,
        number_of_rows, number_of_faces,
    )
    normal = sparse(
        normal_rows, normal_columns, normal_values,
        number_of_rows, number_of_faces,
    )
    return (;
        tangent, normal, cell_indices = retained_cells,
        points = retained_points, normals = retained_normals,
        tangents = retained_tangents, maximum_radius,
    )
end

"""
    build_problem(resolution; ...)

Build the fixed part of the experiment.  The full jump stationarity equation
is

    K_xi * xi + C' * p + N' * lambda = 0.

Define the fixed pressure contribution by

    C' * p = -K_xi * xi_p,

and solve the full equation for `xi`.  Every candidate then uses

    xi[p, lambda] = xi_p - inv(K_xi) * N' * lambda.
"""
function build_problem(
        resolution;
        density = DEFAULT_DENSITY,
        centre = DEFAULT_CENTRE,
        radius = DEFAULT_RADIUS,
        reconstruction_radius = 2.5,
        active_mass_fraction = 1.0e-6,
    )
    T = promote_type(
        typeof(float(density[1])), typeof(float(density[2])),
    )
    rho = (T(density[1]), T(density[2]))
    x0 = (T(centre[1]), T(centre[2]))
    R = T(radius)
    mesh = BlockTreeMesh(
        block_size = (resolution, resolution), periodic = (true, true),
    )
    interface = InterfaceField(
        mesh,
        (x, y) -> hypot(x - x0[1], y - x0[2]) - R;
        lipschitz = one(T),
    )
    solver = TwoPhaseAMR.MimeticProjectionSolver(
        interface; density = rho, jump_interpolant = :cell,
        active_mass_fraction,
    )
    workspace = TwoPhaseAMR.MimeticProjectionWorkspace(solver)
    # Construct only the interface data used by this forward experiment.
    # No elliptic operator, KKT matrix, preconditioner, or right-hand side is
    # needed to apply N, N', inv(K_xi), and T at the interface.
    data = TwoPhaseAMR._mimetic_face_data(solver, workspace)
    active_faces, mass_deactivated, embedded_deactivated =
        TwoPhaseAMR._local_active_faces(solver, data)
    all_face_dual_volumes = isnothing(interface.geometry) ?
        TwoPhaseAMR._uncut_face_dual_volumes(mesh, T) :
        TwoPhaseAMR._geometry_face_dual_volumes(interface.geometry)
    face_dual_volumes = all_face_dual_volumes[active_faces]
    relative_diagonal = data.relative_mass[active_faces] .*
        face_dual_volumes
    incidence = TwoPhaseAMR._mimetic_incidence(mesh, T)
    cell_volumes = fill(prod(mesh.blocks[1].spacing), ncells(mesh))
    assembled = (;
        data, active_faces, relative_diagonal, face_dual_volumes, incidence,
        pressure_operator = (; volumes = cell_volumes),
        mass_deactivated, embedded_deactivated,
    )
    kinetic_diagonal = assembled.relative_diagonal ./
        assembled.face_dual_volumes
    all(>(zero(T)), kinetic_diagonal) ||
        error("the active jump space must have positive kinetic weights")

    aperture_x, aperture_y = TwoPhaseAMR._local_integrated_aperture_vectors(
        interface, assembled.data,
    )
    references = TwoPhaseAMR._cell_references(mesh)
    h = T(mesh.blocks[1].spacing[1])
    cell_indices = Int[]
    points = NTuple{2, T}[]
    normals = NTuple{2, T}[]
    tangents = NTuple{2, T}[]
    for (cell, reference) in pairs(references)
        magnitude = hypot(aperture_x[cell], aperture_y[cell])
        magnitude > T(64) * eps(T) * h || continue
        centre_c = cellcenter(mesh, reference.block, reference.i, reference.j)
        point, normal, tangent = circle_point_and_frame(centre_c, x0, R)
        push!(cell_indices, cell)
        push!(points, T.(point))
        push!(normals, T.(normal))
        push!(tangents, T.(tangent))
    end
    reconstruction = fixed_tangent_reconstruction(
        mesh, active_faces, cell_indices, points, normals, tangents;
        reconstruction_radius = T(reconstruction_radius),
    )
    isempty(reconstruction.cell_indices) &&
        error("no resolved interface rows were constructed")

    constant_samples = active_constant_samples(mesh, active_faces, T)
    constant_error = maximum(abs,
        reconstruction.tangent * constant_samples -
            reduce(vcat, (reshape(collect(t), 1, 2) for t in reconstruction.tangents)),
    )
    constant_error <= T(256) * eps(T) ||
        error("the fixed tangential interpolant does not reproduce constants")

    pressure_potential = Vector{T}(undef, length(references))
    for (cell, reference) in pairs(references)
        x, y = cellcenter(mesh, reference.block, reference.i, reference.j)
        pressure_potential[cell] =
            phase_pressure(2, x, y) / rho[2] -
            phase_pressure(1, x, y) / rho[1]
    end
    pressure_jump_gradient = (
        assembled.incidence * pressure_potential
    )[active_faces] ./ assembled.face_dual_volumes
    target_tangent = T[
        dot2(
            tangent,
            pressure_potential_gradient(rho, point[1], point[2]),
        )
            for (point, tangent) in
                zip(reconstruction.points, reconstruction.tangents)
    ]
    return CommutativityProblem(
        resolution, mesh, interface, assembled, rho, x0, R, active_faces,
        kinetic_diagonal, constant_samples, pressure_jump_gradient,
        reconstruction.tangent, reconstruction.normal, target_tangent,
        reconstruction.cell_indices, reconstruction.points,
        reconstruction.normals, reconstruction.tangents,
        T(reconstruction.maximum_radius), T(active_mass_fraction),
    )
end

function row_geometry(problem, locations, method)
    T = eltype(problem.kinetic_diagonal)
    points = NTuple{2, T}[]
    normals = NTuple{2, T}[]
    if method === :face
        for face_index in locations
            centre_f = problem.mesh.faces[face_index].center
            point, normal, = circle_point_and_frame(
                centre_f, problem.centre, problem.radius,
            )
            push!(points, T.(point))
            push!(normals, T.(normal))
        end
    else
        references = TwoPhaseAMR._cell_references(problem.mesh)
        for cell in locations
            reference = references[cell]
            centre_c = cellcenter(
                problem.mesh, reference.block, reference.i, reference.j,
            )
            point, normal, = circle_point_and_frame(
                centre_c, problem.centre, problem.radius,
            )
            push!(points, T.(point))
            push!(normals, T.(normal))
        end
    end
    return points, normals
end

function normalise_existing_rows(raw, constant_samples, normals)
    response = Matrix(raw * constant_samples)
    retained = Int[]
    scales = eltype(response)[]
    tolerance = 256eps(eltype(response))
    for row in axes(response, 1)
        projection = response[row, 1] * normals[row][1] +
            response[row, 2] * normals[row][2]
        abs(projection) > tolerance || continue
        push!(retained, row)
        push!(scales, inv(projection))
    end
    normal = spdiagm(0 => scales) * raw[retained, :]
    return normal, retained
end

"""Use one of the three normal interpolants already implemented in production."""
function existing_normal(problem, method::Symbol)
    method in (:divergence, :face, :cell) ||
        throw(ArgumentError("unknown existing normal method: $method"))
    solver = TwoPhaseAMR.MimeticProjectionSolver(
        problem.interface; density = problem.density,
        jump_interpolant = method,
        active_mass_fraction = problem.active_mass_fraction,
    )
    operator = TwoPhaseAMR._local_constraint_operator(
        solver, problem.assembly.data, problem.active_faces,
        problem.assembly.relative_diagonal, problem.assembly.incidence,
        problem.assembly.pressure_operator,
    )
    points, normals = row_geometry(
        problem, operator.constraint_locations, method,
    )
    normal, retained = normalise_existing_rows(
        operator.constraint, problem.constant_samples, normals,
    )
    points = points[retained]
    normals = normals[retained]
    supports = [
        count(!iszero, view(normal, row, :)) for row in axes(normal, 1)
    ]
    return NormalCandidate(
        method, normal, points, normals, zero(eltype(normal)),
        isempty(supports) ? zero(eltype(normal)) :
            sum(supports) / length(supports),
        maximum(supports; init = 0),
    )
end

function psd_solve(matrix, right_hand_side; relative_tolerance = 1.0e-11)
    T = eltype(matrix)
    isempty(matrix) && return zeros(T, size(matrix, 2)), 0, zero(T)
    factorisation = eigen(Symmetric(Matrix(matrix)))
    largest = maximum(abs, factorisation.values; init = zero(T))
    cutoff = T(relative_tolerance) * max(largest, one(T))
    inverse_values = map(factorisation.values) do value
        value > cutoff ? inv(value) : zero(T)
    end
    rank = count(>(cutoff), factorisation.values)
    solution = factorisation.vectors *
        (inverse_values .* (transpose(factorisation.vectors) * right_hand_side))
    residual = norm(matrix * solution - right_hand_side, Inf)
    return solution, rank, residual
end

function constrained_patch_projection(
        baseline, constraints, target, kinetic;
        relative_tolerance = 1.0e-11,
    )
    T = eltype(baseline)
    inverse_metric = kinetic
    scales = T[]
    retained = Int[]
    for row in axes(constraints, 1)
        scale = sqrt(sum(
            abs2(constraints[row, column]) * inverse_metric[column]
                for column in axes(constraints, 2)
        ))
        if scale > T(64) * eps(T)
            push!(retained, row)
            push!(scales, scale)
        elseif abs(target[row]) > T(64) * eps(T)
            return baseline, T(Inf)
        end
    end
    isempty(retained) && return baseline, zero(T)
    A = Matrix(constraints[retained, :])
    d = target[retained]
    for row in axes(A, 1)
        A[row, :] ./= scales[row]
        d[row] /= scales[row]
    end
    gram = (A .* transpose(inverse_metric)) * transpose(A)
    correction_rhs = A * baseline - d
    multiplier, _, _ = psd_solve(
        gram, correction_rhs; relative_tolerance,
    )
    result = baseline .- inverse_metric .* (transpose(A) * multiplier)
    residual = norm(A * result - d, Inf)
    return result, residual
end

function periodic_cell_distance(reference_a, reference_b, resolution)
    di = abs(reference_a.i - reference_b.i)
    dj = abs(reference_a.j - reference_b.j)
    return max(min(di, resolution - di), min(dj, resolution - dj))
end

"""
    commuting_normal(problem; patch_radius=2)

Construct `Q=N'` column by column.  Each local column is the minimum kinetic
change from the centred aperture-normal baseline subject to

    T * inv(K_xi) * q_r = 0,
    Z' * q_r = n_r,

where `Z` contains the two constant vector fields.  The second condition is
the first-order normal consistency moment.
"""
function commuting_normal(
        problem;
        patch_radius = 2,
        relative_tolerance = 1.0e-11,
    )
    T = eltype(problem.kinetic_diagonal)
    references = TwoPhaseAMR._cell_references(problem.mesh)
    tangent = problem.tangent
    baseline = problem.normal_baseline
    kinetic_inverse = inv.(problem.kinetic_diagonal)
    q_rows = Int[]
    q_columns = Int[]
    q_values = T[]
    maximum_residual = zero(T)
    supports = Int[]
    for normal_row in axes(baseline, 1)
        reference = references[problem.row_cells[normal_row]]
        nearby_rows = findall(eachindex(problem.row_cells)) do tangent_row
            other = references[problem.row_cells[tangent_row]]
            periodic_cell_distance(
                reference, other, problem.resolution,
            ) <= patch_radius
        end
        support = Int[]
        for tangent_row in nearby_rows
            append!(support, findall(!iszero, view(tangent, tangent_row, :)))
        end
        append!(support, findall(!iszero, view(baseline, normal_row, :)))
        sort!(unique!(support))
        overlapping_rows = findall(axes(tangent, 1)) do tangent_row
            !iszero(nnz(tangent[tangent_row:tangent_row, support]))
        end
        tangential_constraints = Matrix(tangent[overlapping_rows, support])
        tangential_constraints .*= transpose(kinetic_inverse[support])
        moment_constraints = transpose(problem.constant_samples[support, :])
        constraints = vcat(tangential_constraints, moment_constraints)
        target = vcat(
            zeros(T, length(overlapping_rows)),
            collect(problem.row_normals[normal_row]),
        )
        q0 = vec(Array(baseline[normal_row:normal_row, support]))
        local_q, residual = constrained_patch_projection(
            q0, constraints, target, problem.kinetic_diagonal[support];
            relative_tolerance,
        )
        maximum_residual = max(maximum_residual, residual)
        push!(supports, count(!iszero, local_q))
        for (local_index, value) in pairs(local_q)
            abs(value) > T(64) * eps(T) || continue
            push!(q_rows, support[local_index])
            push!(q_columns, normal_row)
            push!(q_values, value)
        end
    end
    Q = sparse(
        q_rows, q_columns, q_values,
        length(problem.active_faces), size(baseline, 1),
    )
    normal = sparse(transpose(Q))
    return NormalCandidate(
        Symbol("commuting_r", patch_radius), normal,
        problem.row_points, problem.row_normals, maximum_residual,
        isempty(supports) ? zero(T) : sum(supports) / length(supports),
        maximum(supports; init = 0),
    )
end

function centred_normal(problem)
    supports = [
        count(!iszero, view(problem.normal_baseline, row, :))
            for row in axes(problem.normal_baseline, 1)
    ]
    return NormalCandidate(
        :centred, problem.normal_baseline, problem.row_points,
        problem.row_normals, zero(eltype(problem.kinetic_diagonal)),
        sum(supports) / length(supports), maximum(supports),
    )
end

function smooth_normal_error(problem, candidate)
    T = eltype(problem.kinetic_diagonal)
    samples = Vector{T}(undef, length(problem.active_faces))
    for (local_face, global_face) in pairs(problem.active_faces)
        face = problem.mesh.faces[global_face]
        vector = manufactured_jump(face.center[1], face.center[2])
        samples[local_face] = vector[face.axis]
    end
    target = T[
        dot2(normal, manufactured_jump(point[1], point[2]))
            for (point, normal) in zip(candidate.points, candidate.normals)
    ]
    error = candidate.matrix * samples - target
    return sqrt(sum(abs2, error) / max(length(error), 1)),
        maximum(abs, error; init = zero(T))
end

@inline function multiplier_probe(point, centre)
    angle = atan(point[2] - centre[2], point[1] - centre[1])
    return sin(3angle) + 0.25cos(2angle)
end

function analyse(problem, candidate)
    T = eltype(problem.kinetic_diagonal)
    N = candidate.matrix
    target_constants = reduce(
        vcat,
        (reshape(collect(normal), 1, 2) for normal in candidate.normals),
    )
    constant_error = maximum(
        abs, N * problem.constant_samples - target_constants;
        init = zero(T),
    )
    normal_l2, normal_linf = smooth_normal_error(problem, candidate)
    pressure_error = problem.tangent * problem.pressure_jump_gradient .-
        problem.target_tangent
    pressure_l2 = sqrt(sum(abs2, pressure_error) / length(pressure_error))
    pressure_linf = maximum(abs, pressure_error)

    multiplier = T[
        multiplier_probe(point, problem.centre) for point in candidate.points
    ]
    # This is deliberately a forward application of the interface operators:
    # no Schur solve and no explicit T * inv(K_xi) * N' matrix are formed.
    multiplier_correction = (transpose(N) * multiplier) ./
        problem.kinetic_diagonal
    multiplier_error = problem.tangent * multiplier_correction
    xi = problem.pressure_jump_gradient .- multiplier_correction
    combined_error = problem.tangent * xi .- problem.target_tangent
    multiplier_l2 = sqrt(
        sum(abs2, multiplier_error) / length(multiplier_error),
    )
    multiplier_linf = maximum(abs, multiplier_error; init = zero(T))
    combined_l2 = sqrt(sum(abs2, combined_error) / length(combined_error))
    combined_linf = maximum(abs, combined_error)
    return CandidateDiagnostics(
        problem.resolution, candidate.name, size(N, 1), constant_error,
        normal_l2, normal_linf, pressure_l2, pressure_linf,
        combined_l2, combined_linf,
        multiplier_l2, multiplier_linf, candidate.construction_residual,
        candidate.mean_support, candidate.maximum_support,
    )
end

function convergence_rate(coarse_error, fine_error, coarse_n, fine_n)
    coarse_error > 0 && fine_error > 0 || return Inf
    return log(coarse_error / fine_error) / log(fine_n / coarse_n)
end

function convergence_rate_label(coarse_error, fine_error, coarse_n, fine_n)
    max(abs(coarse_error), abs(fine_error)) <= 1.0e-11 && return "roundoff"
    return @sprintf(
        "%.3f",
        convergence_rate(coarse_error, fine_error, coarse_n, fine_n),
    )
end

function print_diagnostics(results)
    println("\nConsistency errors (normalised discrete L2 and Linf)")
    @printf(
        "%-5s %-15s %12s %12s %12s %12s %12s %12s\n",
        "n", "N", "normal L2", "normal Linf", "p L2", "p Linf",
        "lambda L2", "lambda Linf",
    )
    for result in results
        @printf(
            "%-5d %-15s %12.3e %12.3e %12.3e %12.3e %12.3e %12.3e\n",
            result.resolution, String(result.name),
            result.normal_l2_error, result.normal_linf_error,
            result.pressure_l2_error, result.pressure_linf_error,
            result.multiplier_l2_error, result.multiplier_linf_error,
        )
    end
    println("\nEndpoint rates over the full resolution range")
    @printf(
        "%-15s %12s %12s %12s %12s %12s %12s\n",
        "N", "normal L2", "normal Linf", "p L2", "p Linf",
        "lambda L2", "lambda Linf",
    )
    names = unique(result.name for result in results)
    for name in names
        selected = sort(
            filter(result -> result.name == name, results);
            by = result -> result.resolution,
        )
        length(selected) >= 2 || continue
        coarse = selected[1]
        fine = selected[end]
        @printf(
            "%-15s %12s %12s %12s %12s %12s %12s\n",
            String(name),
            convergence_rate_label(
                coarse.normal_l2_error, fine.normal_l2_error,
                coarse.resolution, fine.resolution,
            ),
            convergence_rate_label(
                coarse.normal_linf_error, fine.normal_linf_error,
                coarse.resolution, fine.resolution,
            ),
            convergence_rate_label(
                coarse.pressure_l2_error, fine.pressure_l2_error,
                coarse.resolution, fine.resolution,
            ),
            convergence_rate_label(
                coarse.pressure_linf_error, fine.pressure_linf_error,
                coarse.resolution, fine.resolution,
            ),
            convergence_rate_label(
                coarse.multiplier_l2_error, fine.multiplier_l2_error,
                coarse.resolution, fine.resolution,
            ),
            convergence_rate_label(
                coarse.multiplier_linf_error, fine.multiplier_linf_error,
                coarse.resolution, fine.resolution,
            ),
        )
    end
    return nothing
end

"""
    run_study(; resolutions=(16, 24, 32, 48), builders)

Run the cheap refinement study.  A builder is any `problem ->
NormalCandidate` function, so experimental interpolants can be added without
changing the geometry, pressure data, `T`, or diagnostics.
"""
function run_study(;
        resolutions = (16, 24, 32, 48),
        density = DEFAULT_DENSITY,
        reconstruction_radius = 2.5,
        builders,
    )
    results = CandidateDiagnostics[]
    for resolution in resolutions
        problem = build_problem(
            resolution; density, reconstruction_radius,
        )
        @printf(
            "n=%d: %d active faces, %d T rows, reconstruction radius %.2fh\n",
            resolution, length(problem.active_faces),
            size(problem.tangent, 1), problem.reconstruction_radius,
        )
        for builder in builders
            candidate = builder(problem)
            push!(results, analyse(problem, candidate))
        end
    end
    print_diagnostics(results)
    return results
end

"""Forward-only validation of the three normal interpolants in production."""
function run_forward_study(; kwargs...)
    builders = [
        problem -> existing_normal(problem, :divergence),
        problem -> existing_normal(problem, :face),
        problem -> existing_normal(problem, :cell),
    ]
    return run_study(; builders, kwargs...)
end

"""Compare production choices with the experimental commuting construction."""
function run_construction_study(; patch_radius = 2, kwargs...)
    builders = [
        problem -> existing_normal(problem, :divergence),
        problem -> existing_normal(problem, :face),
        problem -> existing_normal(problem, :cell),
        centred_normal,
        problem -> commuting_normal(problem; patch_radius),
    ]
    return run_study(; builders, kwargs...)
end

function main(args = ARGS)
    construct = "--construct" in args
    unknown = filter(argument -> argument != "--construct", args)
    isempty(unknown) || throw(ArgumentError(
        "unknown arguments $(join(unknown, ", ")); expected --construct",
    ))
    started = time_ns()
    construct ? run_construction_study() : run_forward_study()
    @printf(
        "\nStudy computation after module loading: %.3f s\n",
        (time_ns() - started) / 1.0e9,
    )
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module
