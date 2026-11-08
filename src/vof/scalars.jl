"""
    PhaseScalar(interface; liquid=1, gas=1)

Intrinsic phase values `liquid = s_l` and `gas = s_g`. The conservative
products `f_l*s_l` and `(f_open-f_l)*s_g` exist only as temporary terms during
an update; they are never stored in the field. Without embedded geometry,
`f_open=1`. Initialization arguments are
constants or `(x,y)` functions. Advect together with their interface via
`advect!(interface, velocity, dt, work; scalars=(q,))`.
"""
struct PhaseScalar{T, M <: BlockTreeMesh}
    interface::InterfaceField{T, M}
    liquid::CellField{T, M}
    gas::CellField{T, M}
end

"""Interface for selecting how a physical interface is transported."""
abstract type AbstractInterfaceTransportMethod end

"""Internal policy used by the established one-velocity WY path."""
struct _OneVelocityInterfaceAdvection <: AbstractInterfaceTransportMethod end

_interface_transport_workspace(field, ::_OneVelocityInterfaceAdvection) = nothing


"""Conservation diagnostics accumulated from the most recent complete step."""
mutable struct PhaseFluxDiagnostics{T}
    maximum_partition_defect::T
    absolute_partition_defect::T
    signed_partition_defect::T
    gas_reset_discrepancy::T
    absolute_gas_reset_discrepancy::T
    gas_momentum_reset_discrepancy_x::T
    gas_momentum_reset_discrepancy_y::T
    absolute_gas_momentum_reset_discrepancy_x::T
    absolute_gas_momentum_reset_discrepancy_y::T
    projection_initial_divergence::T
    projection_final_divergence::T
    projection_iterations::Int
end

PhaseFluxDiagnostics(::Type{T}) where {T} = PhaseFluxDiagnostics(
    zero(T), zero(T), zero(T), zero(T), zero(T),
    zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), 0,
)

function _reset!(diagnostics::PhaseFluxDiagnostics{T}) where {T}
    diagnostics.maximum_partition_defect = zero(T)
    diagnostics.absolute_partition_defect = zero(T)
    diagnostics.signed_partition_defect = zero(T)
    diagnostics.gas_reset_discrepancy = zero(T)
    diagnostics.absolute_gas_reset_discrepancy = zero(T)
    diagnostics.gas_momentum_reset_discrepancy_x = zero(T)
    diagnostics.gas_momentum_reset_discrepancy_y = zero(T)
    diagnostics.absolute_gas_momentum_reset_discrepancy_x = zero(T)
    diagnostics.absolute_gas_momentum_reset_discrepancy_y = zero(T)
    diagnostics.projection_initial_divergence = zero(T)
    diagnostics.projection_final_divergence = zero(T)
    diagnostics.projection_iterations = 0
    return diagnostics
end

"""
    phase_flux_diagnostics(workspace)

Return the reusable diagnostics object for the most recent completed WY step.
Partition defects have units of volume and compare
`liquid_flux + gas_flux` with `open_flux`. Gas-reset diagnostics are used by
the projected-liquid method to report the signed sum and sum of absolute
differences between its directional auxiliary gas carriers and the physical
complement of the liquid fraction. When separate phase momentum is advanced,
the component-wise momentum-reset fields report the corresponding signed and
absolute changes after multiplication by gas density and dual volume.
"""
phase_flux_diagnostics(workspace) = workspace.diagnostics

_uses_auxiliary_gas_carrier(velocity, workspace) = false
function PhaseScalar(interface::InterfaceField{T}; liquid = 1, gas = 1) where {T}
    mesh = interface.mesh
    q = PhaseScalar(interface, CellField(mesh, T), CellField(mesh, T))
    nx, _ = mesh.block_size
    h = mesh.halo
    _parallel_block_columns(mesh) do b, first_j, last_j
        liquid_values = @inbounds q.liquid.blocks[b]
        gas_values = @inbounds q.gas.blocks[b]
        block = @inbounds mesh.blocks[b]
        dx, dy = block.spacing
        @inbounds for j in first_j:last_j, i in 1:nx
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
