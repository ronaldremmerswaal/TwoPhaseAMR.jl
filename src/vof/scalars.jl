"""
    PhaseScalar(interface; liquid=1, gas=1)

Intrinsic phase values `liquid = s_l` and `gas = s_g`. The conservative
products `f_l*s_l` and `(f_open-f_l)*s_g` exist only as temporary terms during
an update; they are never stored in the field. Without embedded geometry,
`f_open=1`. Initialization arguments are
constants or `(x,y)` functions. Advect together with their interface via
`advect!(interface, velocity, dt, work; scalars=(q,))`.
"""
struct PhaseScalar{T, M <: BlockTreeMesh, G}
    interface::InterfaceField{T, M, G}
    liquid::CellField{T, M}
    gas::CellField{T, M}
end

"""Conservation diagnostics accumulated from the most recent complete step."""
mutable struct PhaseFluxDiagnostics{T}
    maximum_partition_defect::T
    absolute_partition_defect::T
    signed_partition_defect::T
end

PhaseFluxDiagnostics(::Type{T}) where {T} =
    PhaseFluxDiagnostics(zero(T), zero(T), zero(T))

function _reset!(diagnostics::PhaseFluxDiagnostics{T}) where {T}
    diagnostics.maximum_partition_defect = zero(T)
    diagnostics.absolute_partition_defect = zero(T)
    diagnostics.signed_partition_defect = zero(T)
    return diagnostics
end

"""
    phase_flux_diagnostics(workspace)

Return the reusable diagnostics object for the most recent completed WY step.
Partition defects have units of volume and compare
`liquid_flux + gas_flux` with `open_flux`.
"""
phase_flux_diagnostics(workspace) = workspace.diagnostics

function PhaseScalar(interface::InterfaceField{T}; liquid = 1, gas = 1) where {T}
    mesh = interface.mesh
    q = PhaseScalar(interface, CellField(mesh, T), CellField(mesh, T))
    nx, ny = mesh.block_size
    h = mesh.halo
    # Serial: `liquid` and `gas` may be user callbacks.
    for (b, block) in pairs(mesh.blocks)
        liquid_values = @inbounds q.liquid.blocks[b]
        gas_values = @inbounds q.gas.blocks[b]
        dx, dy = block.spacing
        @inbounds for j in 1:ny, i in 1:nx
            x = block.origin[1] + (i - 1 / 2) * dx
            y = block.origin[2] + (j - 1 / 2) * dy
            l = liquid isa Number ? liquid : liquid(x, y)
            g = gas isa Number ? gas : gas(x, y)
            isfinite(l) && isfinite(g) ||
                throw(ArgumentError("phase scalar values must be finite"))
            liquid_values[i + h, j + h] = l
            gas_values[i + h, j + h] = g
        end
    end
    return q
end
