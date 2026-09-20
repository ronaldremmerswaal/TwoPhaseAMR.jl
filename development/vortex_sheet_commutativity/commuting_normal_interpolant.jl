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
    pressure_stationarity_jump::Vector{T}
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
    tangent_l2_error::T
    tangent_linf_error::T
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

@inline function signed_periodic_delta(a, b)
    delta = a - b
    delta > 0.5 && (delta -= 1)
    delta < -0.5 && (delta += 1)
    return delta
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

@inline function periodic_bucket(index, resolution)
    return mod(index - 1, resolution) + 1
end

@inline function point_bucket(point, h, resolution)
    return (
        periodic_bucket(floor(Int, point[1] / h) + 1, resolution),
        periodic_bucket(floor(Int, point[2] / h) + 1, resolution),
    )
end

function active_face_buckets(mesh, active_faces, h, resolution)
    buckets = ntuple(_ -> Dict{Tuple{Int, Int}, Vector{Int}}(), 2)
    for (local_face, global_face) in pairs(active_faces)
        face = mesh.faces[global_face]
        key = point_bucket(face.center, h, resolution)
        push!(get!(buckets[face.axis], key, Int[]), local_face)
    end
    return buckets
end

function nearby_axis_faces(
        mesh, active_faces, buckets, point, axis, h, resolution,
        initial_radius,
    )
    centre_bucket = point_bucket(point, h, resolution)
    for radius in initial_radius:one(initial_radius):initial_radius + 4
        candidates = Tuple{Int, Float64}[]
        limit = radius * h
        bucket_radius = ceil(Int, radius) + 1
        for dj in -bucket_radius:bucket_radius
            for di in -bucket_radius:bucket_radius
                key = (
                    periodic_bucket(centre_bucket[1] + di, resolution),
                    periodic_bucket(centre_bucket[2] + dj, resolution),
                )
                for local_face in get(buckets[axis], key, Int[])
                    face = mesh.faces[active_faces[local_face]]
                    dx = periodic_delta(face.center[1], point[1])
                    dy = periodic_delta(face.center[2], point[2])
                    max(dx, dy) <= limit || continue
                    push!(candidates, (local_face, hypot(dx, dy)))
                end
            end
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
    resolution = mesh.block_size[1]
    buckets = active_face_buckets(mesh, active_faces, h, resolution)
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
                mesh, active_faces, buckets, points[row], axis, h,
                resolution,
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

    # Sample the physical, piecewise pressure represented by the single cell
    # pressure unknown in the production projection.  The default phase
    # pressures have equal traces (and are in fact equal everywhere), so no
    # separate pressure-jump lifting is needed in this focused experiment.
    pressure_values = Vector{T}(undef, length(references))
    for (cell, reference) in pairs(references)
        x, y = cellcenter(mesh, reference.block, reference.i, reference.j)
        phase = hypot(x - x0[1], y - x0[2]) <= R ? 1 : 2
        pressure_values[cell] = phase_pressure(phase, x, y)
    end
    # This is the actual pressure part of the production stationarity
    # equation, not a separately manufactured compatible gradient:
    #
    #   K_xi xi_p + C' p = 0,
    #   C' p = -diag(coupling) I p,
    #   xi_p = -inv(K_xi) C' p.
    #
    # Here K_xi is `relative_diagonal` and I is the integrated incidence.
    pressure_stationarity_jump =
        data.coupling[active_faces] .*
        (assembled.incidence * pressure_values)[active_faces] ./
        assembled.relative_diagonal
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
        kinetic_diagonal, constant_samples, pressure_stationarity_jump,
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

function sparse_row_supports(matrix)
    supports = [Int[] for _ in axes(matrix, 1)]
    rows, columns, = findnz(matrix)
    for (row, column) in zip(rows, columns)
        push!(supports[row], column)
    end
    return supports
end

function sparse_column_rows(row_supports, number_of_columns)
    rows = [Int[] for _ in 1:number_of_columns]
    for (row, support) in pairs(row_supports)
        for column in support
            push!(rows[column], row)
        end
    end
    return rows
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
    tangent_supports = sparse_row_supports(tangent)
    baseline_supports = sparse_row_supports(baseline)
    tangent_rows_at_face = sparse_column_rows(
        tangent_supports, size(tangent, 2),
    )
    row_lookup = Dict{Tuple{Int, Int}, Int}()
    for (row, cell) in pairs(problem.row_cells)
        reference = references[cell]
        row_lookup[(reference.i, reference.j)] = row
    end
    q_rows = Int[]
    q_columns = Int[]
    q_values = T[]
    maximum_residual = zero(T)
    supports = Int[]
    for normal_row in axes(baseline, 1)
        reference = references[problem.row_cells[normal_row]]
        nearby_rows = Int[]
        for dj in -patch_radius:patch_radius
            for di in -patch_radius:patch_radius
                key = (
                    periodic_bucket(reference.i + di, problem.resolution),
                    periodic_bucket(reference.j + dj, problem.resolution),
                )
                tangent_row = get(row_lookup, key, 0)
                tangent_row > 0 && push!(nearby_rows, tangent_row)
            end
        end
        support = Int[]
        for tangent_row in nearby_rows
            append!(support, tangent_supports[tangent_row])
        end
        append!(support, baseline_supports[normal_row])
        sort!(unique!(support))
        overlapping_rows = Int[]
        for face in support
            append!(overlapping_rows, tangent_rows_at_face[face])
        end
        sort!(unique!(overlapping_rows))
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

function with_tangent(problem::CommutativityProblem{T}, tangent) where {T}
    return CommutativityProblem(
        problem.resolution, problem.mesh, problem.interface,
        problem.assembly, problem.density, problem.centre, problem.radius,
        problem.active_faces, problem.kinetic_diagonal,
        problem.constant_samples, problem.pressure_stationarity_jump,
        sparse(T.(tangent)), problem.normal_baseline,
        problem.target_tangent, problem.row_cells, problem.row_points,
        problem.row_normals, problem.row_tangents,
        problem.reconstruction_radius, problem.active_mass_fraction,
    )
end

function linear_pressure_response(problem, axis)
    T = eltype(problem.kinetic_diagonal)
    references = TwoPhaseAMR._cell_references(problem.mesh)
    pressure = Vector{T}(undef, length(references))
    for (cell, reference) in pairs(references)
        point = cellcenter(
            problem.mesh, reference.block, reference.i, reference.j,
        )
        pressure[cell] = point[axis]
    end
    data = problem.assembly.data
    return data.coupling[problem.active_faces] .*
        (problem.assembly.incidence * pressure)[problem.active_faces] ./
        problem.assembly.relative_diagonal
end

"""
    pressure_compatible_tangent(problem)

Construct a local tangential observation before constructing `N`.  Every
row makes the smallest Euclidean change from the geometric reconstruction
subject to one exact and one asymptotic set of moments:

1. affine Cartesian vector fields are observed at the interface point;
2. the production pressure stationarity response is fitted for the affine
   pressures `p=x` and `p=y` with an `O(h)` consistency target.

The second condition is independent of the smooth trigonometric pressure
used by [`analyse`](@ref), so convergence of the reported pressure error is
an out-of-sample consistency check rather than a fit to that error vector.
Compact support and bounded row weights, together with the affine moments,
give a consistent tangential interpolation of smooth fields.
"""
function pressure_compatible_tangent(
        problem; relative_tolerance = 1.0e-11,
        regularization_factor = 1, initial_radius = 2.5,
        maximum_radius = 8, radius_step = 1,
        pressure_accuracy_factor = 1, amplification_limit = 4,
    )
    T = eltype(problem.kinetic_diagonal)
    initial_radius = T(initial_radius)
    maximum_radius = T(maximum_radius)
    radius_step = T(radius_step)
    accuracy_factor = T(pressure_accuracy_factor)
    amplification_limit = T(amplification_limit)
    zero(T) < initial_radius <= maximum_radius || throw(ArgumentError(
        "tangent support radii must satisfy 0 < initial <= maximum",
    ))
    radius_step > zero(T) || throw(ArgumentError(
        "tangent radius_step must be positive",
    ))
    accuracy_factor > zero(T) || throw(ArgumentError(
        "pressure_accuracy_factor must be positive",
    ))
    amplification_limit > zero(T) || throw(ArgumentError(
        "tangent amplification_limit must be positive",
    ))
    pressure_x = linear_pressure_response(problem, 1)
    pressure_y = linear_pressure_response(problem, 2)
    inverse_density_jump = inv(problem.density[2]) -
        inv(problem.density[1])
    rows = Int[]
    columns = Int[]
    values = T[]
    maximum_residual = zero(T)
    maximum_amplification = zero(T)
    largest_radius = zero(T)
    fallback_rows = 0
    mean_support = zero(T)
    maximum_support = 0
    h = T(problem.mesh.blocks[1].spacing[1])
    buckets = active_face_buckets(
        problem.mesh, problem.active_faces, h, problem.resolution,
    )
    number_of_rows = length(problem.row_points)
    for row in 1:number_of_rows
        point = problem.row_points[row]
        tangent = problem.row_tangents[row]
        best = nothing
        radius = initial_radius
        while true
            candidates = ntuple(2) do axis
                raw = nearby_axis_faces(
                    problem.mesh, problem.active_faces, buckets, point,
                    axis, h, problem.resolution, radius,
                )
                unique_candidates = Dict{Int, Float64}()
                for (face, distance) in raw
                    unique_candidates[face] = min(
                        get(unique_candidates, face, Inf), distance,
                    )
                end
                sort!(collect(unique_candidates); by = first)
            end
            if !any(isempty, candidates)
                support = sort!(unique!(vcat(
                    first.(candidates[1]), first.(candidates[2]),
                )))
                support_index = Dict(
                    face => local_index
                        for (local_index, face) in pairs(support)
                )
                baseline_row = zeros(T, length(support))
                for axis in 1:2
                    weights = reconstruction_weights(candidates[axis], h)
                    for ((face, _), weight) in zip(
                            candidates[axis], weights,
                        )
                        baseline_row[support_index[face]] +=
                            tangent[axis] * weight
                    end
                end
                moment_constraints = transpose(
                    problem.constant_samples[support, :],
                )
                affine_constraints = zeros(T, 4, length(support))
                for (local_column, active_face) in pairs(support)
                    face = problem.mesh.faces[
                        problem.active_faces[active_face]
                    ]
                    dx = T(signed_periodic_delta(
                        face.center[1], point[1],
                    )) / h
                    dy = T(signed_periodic_delta(
                        face.center[2], point[2],
                    )) / h
                    offset = 2 * (face.axis - 1)
                    affine_constraints[offset + 1, local_column] = dx
                    affine_constraints[offset + 2, local_column] = dy
                end
                consistency_constraints = vcat(
                    moment_constraints, affine_constraints,
                )
                consistency_target = T[
                    tangent[1], tangent[2], 0, 0, 0, 0,
                ]
                consistent_row, consistency_residual =
                    constrained_patch_projection(
                        baseline_row, consistency_constraints,
                        consistency_target, ones(T, length(support));
                        relative_tolerance,
                    )
                pressure_constraints = transpose(hcat(
                    pressure_x[support], pressure_y[support],
                ))
                pressure_target = T[
                    inverse_density_jump * tangent[1],
                    inverse_density_jump * tangent[2],
                ]
                local_row = consistent_row
                consistency_tolerance = T(128) * sqrt(eps(T))
                if consistency_residual <= consistency_tolerance
                    # Search only in the nullspace of the exact geometric
                    # moments.  The O(h^2) Tikhonov term prevents a nearly
                    # singular pressure moment from producing unbounded
                    # interpolation weights.
                    moment_matrix = Matrix(consistency_constraints)
                    scale = max(opnorm(moment_matrix), one(T))
                    null_basis = nullspace(
                        moment_matrix;
                        atol = T(relative_tolerance) * scale,
                    )
                    if !isempty(null_basis)
                        reduced_pressure =
                            pressure_constraints * null_basis
                        pressure_rhs = pressure_target -
                            pressure_constraints * consistent_row
                        alpha = T(regularization_factor) * h^2
                        reduced_matrix = transpose(reduced_pressure) *
                            reduced_pressure + alpha * I
                        reduced_rhs = transpose(reduced_pressure) *
                            pressure_rhs + alpha * transpose(null_basis) *
                            (baseline_row - consistent_row)
                        local_row = consistent_row + null_basis *
                            (reduced_matrix \ reduced_rhs)
                    end
                end
                residual = norm(
                    pressure_constraints * local_row - pressure_target,
                    Inf,
                )
                amplification = norm(local_row) /
                    max(norm(baseline_row), sqrt(eps(T)))
                score = max(
                    consistency_residual / consistency_tolerance,
                    residual / (accuracy_factor * h),
                    amplification / amplification_limit,
                )
                trial = (;
                    support, local_row, residual, amplification, radius,
                    score, consistency_residual,
                )
                (isnothing(best) || trial.score < best.score) &&
                    (best = trial)
                if consistency_residual <= consistency_tolerance &&
                        residual <= accuracy_factor * h &&
                        amplification <= amplification_limit
                    best = trial
                    break
                end
            end
            radius >= maximum_radius && break
            radius = min(radius + radius_step, maximum_radius)
        end
        isnothing(best) && error(
            "no tangential support was found for interface row $row",
        )
        best.score <= one(T) || (fallback_rows += 1)
        support = best.support
        local_row = best.local_row
        residual = best.residual
        maximum_residual = max(maximum_residual, residual)
        maximum_amplification = max(
            maximum_amplification, best.amplification,
        )
        largest_radius = max(largest_radius, best.radius)
        local_support = count(!iszero, local_row)
        mean_support += local_support
        maximum_support = max(maximum_support, local_support)
        for (local_index, value) in pairs(local_row)
            abs(value) > T(64) * eps(T) || continue
            push!(rows, row)
            push!(columns, support[local_index])
            push!(values, value)
        end
    end
    tangent = sparse(
        rows, columns, values, number_of_rows,
        length(problem.active_faces),
    )
    return (;
        tangent, residual = maximum_residual, maximum_amplification,
        mean_support = mean_support / max(number_of_rows, 1),
        maximum_support, maximum_radius = largest_radius, fallback_rows,
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

function smooth_tangent_error(problem)
    T = eltype(problem.kinetic_diagonal)
    samples = Vector{T}(undef, length(problem.active_faces))
    for (local_face, global_face) in pairs(problem.active_faces)
        face = problem.mesh.faces[global_face]
        vector = manufactured_jump(face.center[1], face.center[2])
        samples[local_face] = vector[face.axis]
    end
    target = T[
        dot2(tangent, manufactured_jump(point[1], point[2]))
            for (point, tangent) in
                zip(problem.row_points, problem.row_tangents)
    ]
    error = problem.tangent * samples - target
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
    tangent_l2, tangent_linf = smooth_tangent_error(problem)
    normal_l2, normal_linf = smooth_normal_error(problem, candidate)
    pressure_error =
        problem.tangent * problem.pressure_stationarity_jump .-
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
    xi = problem.pressure_stationarity_jump .- multiplier_correction
    combined_error = problem.tangent * xi .- problem.target_tangent
    multiplier_l2 = sqrt(
        sum(abs2, multiplier_error) / length(multiplier_error),
    )
    multiplier_linf = maximum(abs, multiplier_error; init = zero(T))
    combined_l2 = sqrt(sum(abs2, combined_error) / length(combined_error))
    combined_linf = maximum(abs, combined_error)
    return CandidateDiagnostics(
        problem.resolution, candidate.name, size(N, 1),
        tangent_l2, tangent_linf, constant_error, normal_l2, normal_linf,
        pressure_l2, pressure_linf,
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
    println("\nInterpolation errors (normalised discrete L2 and Linf)")
    @printf(
        "%-5s %-15s %12s %12s %12s %12s\n",
        "n", "(T,N)", "tangent L2", "tangent Linf", "normal L2",
        "normal Linf",
    )
    for result in results
        @printf(
            "%-5d %-15s %12.3e %12.3e %12.3e %12.3e\n",
            result.resolution, String(result.name),
            result.tangent_l2_error, result.tangent_linf_error,
            result.normal_l2_error, result.normal_linf_error,
        )
    end
    println("\nCommutativity errors (normalised discrete L2 and Linf)")
    @printf(
        "%-5s %-15s %12s %12s %12s %12s\n",
        "n", "(T,N)", "p L2", "p Linf", "lambda L2", "lambda Linf",
    )
    for result in results
        @printf(
            "%-5d %-15s %12.3e %12.3e %12.3e %12.3e\n",
            result.resolution, String(result.name),
            result.pressure_l2_error, result.pressure_linf_error,
            result.multiplier_l2_error, result.multiplier_linf_error,
        )
    end
    println("\nEndpoint rates over the full resolution range")
    @printf(
        "%-15s %10s %10s %10s %10s %10s %10s %10s %10s\n",
        "(T,N)", "T L2", "T Linf", "N L2", "N Linf", "p L2",
        "p Linf", "lambda L2", "lambda Linf",
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
            "%-15s %10s %10s %10s %10s %10s %10s %10s %10s\n",
            String(name),
            convergence_rate_label(
                coarse.tangent_l2_error, fine.tangent_l2_error,
                coarse.resolution, fine.resolution,
            ),
            convergence_rate_label(
                coarse.tangent_linf_error, fine.tangent_linf_error,
                coarse.resolution, fine.resolution,
            ),
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

"""Construct `T` from the pressure block, then construct `N'` from `T`."""
function run_pressure_first_study(;
        resolutions = (16, 24, 32, 48), density = DEFAULT_DENSITY,
        reconstruction_radius = 2.5, patch_radius = 4,
    )
    results = CandidateDiagnostics[]
    println(
        "n     T rows  T pressure residual  residual/h  " *
        "T max amplification  radius  fallback  " *
        "N fit residual",
    )
    for resolution in resolutions
        baseline = build_problem(
            resolution; density, reconstruction_radius,
        )
        tangent = pressure_compatible_tangent(baseline)
        problem = with_tangent(baseline, tangent.tangent)
        raw_candidate = commuting_normal(problem; patch_radius)
        candidate = NormalCandidate(
            Symbol("pressure_", raw_candidate.name), raw_candidate.matrix,
            raw_candidate.points, raw_candidate.normals,
            raw_candidate.construction_residual,
            raw_candidate.mean_support, raw_candidate.maximum_support,
        )
        push!(results, analyse(problem, candidate))
        @printf(
            "%-5d %6d %19.3e %11.3e %20.3e %7.1f %9d %15.3e\n",
            resolution, size(problem.tangent, 1), tangent.residual,
            resolution * tangent.residual,
            tangent.maximum_amplification, tangent.maximum_radius,
            tangent.fallback_rows,
            candidate.construction_residual,
        )
    end
    print_diagnostics(results)
    return results
end

"""Report sparse-storage, matvec-work, and construction-cost diagnostics."""
function run_sparsity_study(;
        resolutions = (16, 32, 64, 128), patch_radius = 2, repeats = 5,
    )
    builders = [
        problem -> existing_normal(problem, :divergence),
        problem -> existing_normal(problem, :face),
        problem -> existing_normal(problem, :cell),
        centred_normal,
        problem -> commuting_normal(problem; patch_radius),
    ]
    println(
        "n     N                 rows  cols    nnz  row avg/max  " *
        "col avg/max   build ms",
    )
    for resolution in resolutions
        problem = build_problem(resolution)
        for builder in builders
            candidate = builder(problem) # compile and warm this builder
            timings = Float64[]
            for _ in 1:repeats
                started = time_ns()
                candidate = builder(problem)
                push!(timings, (time_ns() - started) / 1.0e6)
            end
            N = candidate.matrix
            row_counts = length.(sparse_row_supports(N))
            column_counts = diff(N.colptr)
            @printf(
                "%-5d %-16s %5d %5d %6d %6.2f/%-3d %6.2f/%-3d %10.3f\n",
                resolution, String(candidate.name), size(N, 1), size(N, 2),
                nnz(N), sum(row_counts) / max(length(row_counts), 1),
                maximum(row_counts; init = 0),
                sum(column_counts) / max(length(column_counts), 1),
                maximum(column_counts; init = 0), minimum(timings),
            )
        end
    end
    println(
        "\nA single N or N' application uses nnz multiplies and roughly " *
        "nnz additions; sparse storage is O(nnz).",
    )
    return nothing
end

function main(args = ARGS)
    construct = "--construct" in args
    sparsity = "--sparsity" in args
    pressure_first = "--pressure-first" in args
    count((construct, sparsity, pressure_first)) <= 1 || throw(ArgumentError(
        "choose only one of --construct, --sparsity, or --pressure-first",
    ))
    unknown = filter(
        argument -> argument != "--construct" &&
            argument != "--sparsity" && argument != "--pressure-first",
        args,
    )
    isempty(unknown) || throw(ArgumentError(
        "unknown arguments $(join(unknown, ", ")); expected --construct " *
        "--sparsity, or --pressure-first",
    ))
    started = time_ns()
    if sparsity
        run_sparsity_study()
    elseif pressure_first
        run_pressure_first_study()
    elseif construct
        run_construction_study()
    else
        run_forward_study()
    end
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
