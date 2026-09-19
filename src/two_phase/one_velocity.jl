"""Interface for explicit one- or two-velocity flow-model choices."""
abstract type AbstractVelocityModel end

"""Select the shared-velocity two-phase formulation."""
struct OneVelocityModel <: AbstractVelocityModel end
