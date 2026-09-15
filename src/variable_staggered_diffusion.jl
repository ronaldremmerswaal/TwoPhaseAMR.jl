"""Coupled staggered `div(2μD(u))` with cell-centred normal stresses and vertex
shear stresses. The integrated stiffness is `E'WE`, so viscosity dissipates
kinetic energy for positive μ. Density enters the time-step mass matrix.
"""
mutable struct VariableStaggeredDiffusionOperator{T, M, B}
    mesh::M
    stiffness::SparseMatrixCSC{T, Int}
    volumes::Vector{T}
    boundary::B
    strain::SparseMatrixCSC{T, Int}
    stress_interpolation::SparseMatrixCSC{T, Int}
    stress_measures::Vector{T}
    stress_weights::Vector{T}
    cell_work::Vector{T}
    stress_work::Vector{T}
end

@inline function _phase_dynamic_viscosity(c, viscosity::NTuple{2})
    liquid, gas = viscosity
    return exp(muladd(c, log(liquid) - log(gas), log(gas)))
end

"""
    dynamic_viscosity(interface; viscosity)

Return the cell-centred dynamic viscosity used by the one-velocity diffusion
operator. Following the thesis, the coefficient is the phase-fraction-
weighted geometric mean

`mu = mu_liquid^c * mu_gas^(1-c)`.

At shear-stress vertices the operator volume-interpolates `c` before applying
the same formula. Equivalently, it geometrically averages this returned field,
because `log(mu)` is affine in `c`.
"""
function dynamic_viscosity(
        field::InterfaceField{T}; viscosity::NTuple{2}
    ) where {T}
    phases = T.(viscosity)
    all(mu -> isfinite(mu) && mu > zero(T), phases) ||
        throw(
        ArgumentError(
            "phase dynamic viscosities must be finite and positive"
        )
    )
    result = CellField(field.mesh, T)
    for ref in _cell_references(field.mesh)
        _set!(
            result,
            _phase_dynamic_viscosity(_get(field.fraction, ref), phases),
            ref
        )
    end
    return result
end

function _stress_vertices(mesh, ::Type{T}) where {T}
    # Cell-corner quadrature weights partition the domain volume exactly.
    lattice = mesh.block_size .* (1 << mesh.max_level)
    vertices = Dict{NTuple{2, Int}, Tuple{T, Vector{Tuple{CellRef, T}}}}()
    nx, ny = mesh.block_size
    for ref in _cell_references(mesh)
        block = mesh.blocks[ref.block]
        x, y = cellcenter(mesh, ref.block, ref.i, ref.j)
        hx, hy = block.spacing
        for sx in (-1, 1),sy in (-1, 1)
            key = (
                round(Int, (x + sx * hx / 2 - mesh.origin[1]) / mesh.extent[1] * lattice[1]),
                round(Int, (y + sy * hy / 2 - mesh.origin[2]) / mesh.extent[2] * lattice[2]),
            )
            key = ntuple(a -> mesh.periodic[a] ? mod(key[a], lattice[a]) : key[a], 2)
            volume, refs = get!(vertices, key) do
                (zero(T), Tuple{CellRef, T}[])
            end
            weight = T(block.volume / 4)
            push!(refs, (ref, weight))
            vertices[key] = (volume + weight, refs)
        end
    end
    return vertices, lattice
end

function _vertex_derivative(mesh, point, component, scale, boundary, ::Type{T}) where {T}
    axis = 3 - component
    tolerance = 128eps(T) * maximum(mesh.extent)
    points = Tuple{Int, T, T}[]
    for (i, face) in pairs(mesh.faces)
        face.axis == component || continue
        dx = _wrapped_delta(face.center[1] - point[1], mesh.extent[1], mesh.periodic[1])
        dy = _wrapped_delta(face.center[2] - point[2], mesh.extent[2], mesh.periodic[2])
        hypot(dx, dy) <= 3scale && push!(points, (i, T(dx), T(dy)))
    end
    for face in mesh.boundary_faces
        face.axis == component || continue
        dx = face.center[1] - point[1];dy = face.center[2] - point[2]
        hypot(dx, dy) <= 3scale && push!(points, (0, T(dx), T(dy)))
    end
    # Standard MAC shear derivative on regular lines (including wall closure).
    line = filter(p -> abs(p[component + 1]) <= tolerance, points)
    below = filter(p -> p[axis + 1] < -tolerance, line)
    above = filter(p -> p[axis + 1] > tolerance, line)
    wall_low = !mesh.periodic[axis] && abs(point[axis] - mesh.origin[axis]) <= tolerance
    wall_high = !mesh.periodic[axis] && abs(point[axis] - mesh.origin[axis] - mesh.extent[axis]) <= tolerance
    if (wall_low || wall_high) && boundary isa FreeSlip
        return Int[], T[]
    end
    if wall_low
        push!(below, (0, zero(T), zero(T)))
    elseif wall_high
        push!(above, (0, zero(T), zero(T)))
    end
    if !isempty(below) && !isempty(above)
        low = below[argmax(p[axis + 1] for p in below)]
        high = above[argmin(p[axis + 1] for p in above)]
        distance = high[axis + 1] - low[axis + 1]
        ids = Int[];weights = T[]
        low[1] > 0 && (push!(ids, low[1]); push!(weights, -inv(distance)))
        high[1] > 0 && (push!(ids, high[1]); push!(weights, inv(distance)))
        return ids, weights
    end
    # At hanging vertices a collocated pair can be absent. A weighted quadratic
    # fit reproduces affine strain while using the same owned face unknowns.
    sort!(points; by = p -> p[2]^2 + p[3]^2)
    length(points) > 16 && resize!(points, 16)
    for degree in (2, 1)
        columns = degree == 2 ? 6 : 3
        length(points) >= columns || continue
        matrix = Matrix{T}(undef, length(points), columns)
        weights = Vector{T}(undef, length(points))
        for (i, p) in pairs(points)
            x, y = p[2] / scale, p[3] / scale
            weights[i] = inv(1 + x * x + y * y)
            row = degree == 2 ? (one(T), x, y, x * x, x * y, y * y) : (one(T), x, y)
            for j in 1:columns
                matrix[i, j] = weights[i] * row[j]
            end
        end
        singular = svdvals(matrix)
        minimum(singular) > 1.0e-10maximum(singular) || continue
        target = zeros(T, columns);target[axis + 1] = inv(scale)
        # Minimal-norm solution of the polynomial reproduction constraints.
        coefficients = transpose(matrix) \ target
        ids = Int[];stencil = T[]
        for (i, p) in pairs(points)
            p[1] > 0 || continue
            push!(ids, p[1]);push!(stencil, weights[i] * coefficients[i])
        end
        return ids, stencil
    end
    throw(ArgumentError("insufficient faces for staggered shear reconstruction"))
end

function _variable_staggered_diffusion(mesh, viscosity::CellField, boundary, ::Type{T}) where {T}
    viscosity.mesh === mesh || throw(ArgumentError("viscosity field must share the operator mesh"))
    boundary isa FreeSlip || (boundary isa NoSlip && boundary.velocity isa StationaryWall) ||
        throw(ArgumentError("variable stress diffusion currently supports stationary NoSlip or FreeSlip walls"))
    all(c -> isfinite(c)&&c > 0, (v for b in eachindex(mesh.blocks) for v in interior(viscosity, b))) ||
        throw(ArgumentError("dynamic viscosity must be finite and positive"))
    refs = _cell_references(mesh);nx, ny = mesh.block_size
    rows = Int[];cols = Int[];vals = T[]
    interpolation_rows = Int[]
    interpolation_columns = Int[]
    interpolation_values = T[]
    measures = T[]
    for ref in refs,axis in 1:2
        row = length(measures) + 1
        push!(measures, 2 * mesh.blocks[ref.block].volume)
        push!(interpolation_rows, row)
        push!(interpolation_columns, _cell_index(ref, nx, ny))
        push!(interpolation_values, one(T))
    end
    for (i, face) in pairs(mesh.faces)
        for (ref, sign) in ((face.minus, 1), (face.plus, -1))
            row = 2 * (_cell_index(ref, nx, ny) - 1) + face.axis
            push!(rows, row);push!(cols, i)
            push!(vals, sign * face.area / mesh.blocks[ref.block].volume)
        end
    end
    vertices, lattice = _stress_vertices(mesh, T)
    for key in sort!(collect(keys(vertices)))
        volume, cells = vertices[key]
        point = ntuple(a -> mesh.origin[a] + key[a] * mesh.extent[a] / lattice[a], 2)
        push!(measures, volume);row = length(measures)
        for (ref, weight) in cells
            push!(interpolation_rows, row)
            push!(interpolation_columns, _cell_index(ref, nx, ny))
            push!(interpolation_values, weight / volume)
        end
        scale = maximum(maximum(mesh.blocks[r.block].spacing) for (r, w) in cells)
        for component in 1:2
            indices, coefficients = _vertex_derivative(mesh, point, component, scale, boundary, T)
            append!(rows, fill(row, length(indices)));append!(cols, indices);append!(vals, coefficients)
        end
    end
    strain = sparse(rows, cols, vals, length(measures), length(mesh.faces))
    interpolation = sparse(
        interpolation_rows, interpolation_columns, interpolation_values,
        length(measures), length(refs)
    )
    cell_work = T[log(_get(viscosity, ref)) for ref in refs]
    stress_work = interpolation * cell_work
    weights = measures .* exp.(stress_work)
    stiffness = transpose(strain) * spdiagm(0 => weights) * strain
    volumes = T[face.area * (mesh.blocks[face.minus.block].spacing[face.axis] + mesh.blocks[face.plus.block].spacing[face.axis]) / 2 for face in mesh.faces]
    return VariableStaggeredDiffusionOperator(
        mesh, stiffness, volumes, boundary, strain, interpolation,
        measures, weights, cell_work, stress_work
    )
end

"""Refresh the stress-location geometric mean without rebuilding geometric
strain and interpolation stencils. This is used after every VOF update."""
function _update_dynamic_viscosity!(
        operator::VariableStaggeredDiffusionOperator{T},
        field::InterfaceField{T}, viscosity::NTuple{2}
    ) where {T}
    operator.mesh === field.mesh ||
        throw(ArgumentError("interface and diffusion operator must share a mesh"))
    phases = T.(viscosity)
    all(mu -> isfinite(mu) && mu > zero(T), phases) ||
        throw(
        ArgumentError(
            "phase dynamic viscosities must be finite and positive"
        )
    )
    log_gas = log(phases[2])
    log_jump = log(phases[1]) - log_gas
    for (index, ref) in pairs(_cell_references(field.mesh))
        @inbounds operator.cell_work[index] =
            muladd(_get(field.fraction, ref), log_jump, log_gas)
    end
    mul!(
        operator.stress_work, operator.stress_interpolation,
        operator.cell_work
    )
    for index in eachindex(operator.stress_weights)
        @inbounds operator.stress_weights[index] =
            operator.stress_measures[index] * exp(operator.stress_work[index])
    end
    operator.stiffness = transpose(operator.strain) *
        spdiagm(0 => operator.stress_weights) * operator.strain
    return operator
end

function StaggeredDiffusionOperator(field::InterfaceField; viscosity::NTuple{2}, boundary = NoSlip())
    mu = dynamic_viscosity(field; viscosity)
    return StaggeredDiffusionOperator(field.mesh; viscosity = mu, boundary)
end
function diffusion!(
        destination::FaceVelocity, velocity::FaceVelocity,
        operator::VariableStaggeredDiffusionOperator; time = 0
    )
    destination.mesh === velocity.mesh === operator.mesh || throw(ArgumentError("diffusion arguments must share a mesh"))
    destination === velocity && throw(ArgumentError("diffusion output must not alias input"))
    mul!(destination.interior, operator.stiffness, velocity.interior)
    destination.interior ./= -operator.volumes
    fill!(destination.boundary, zero(eltype(destination)))
    return destination
end
"""Implicit midpoint of the coupled stress operator, with an optional positive
staggered density. The sparse solve is rebuilt here when coefficients change."""
function implicit_midpoint_diffusion!(
        destination::FaceVelocity, source::FaceVelocity,
        operator::VariableStaggeredDiffusionOperator, dt; density = ones(eltype(source), length(source.interior))
    )
    destination.mesh === source.mesh === operator.mesh || throw(ArgumentError("diffusion arguments must share a mesh"))
    isfinite(dt)&&dt > 0 || throw(ArgumentError("dt must be finite and positive"))
    length(density) == length(operator.volumes) || throw(DimensionMismatch("one density per staggered volume is required"))
    all(rho -> isfinite(rho)&&rho > 0, density) || throw(ArgumentError("density must be finite and positive"))
    inertia = density .* operator.volumes
    matrix = spdiagm(0 => inertia) + (dt / 2) * operator.stiffness
    rhs = inertia .* source.interior - (dt / 2) * (operator.stiffness * source.interior)
    destination.interior .= matrix \ rhs
    fill!(destination.boundary, zero(eltype(destination)))
    return destination
end
