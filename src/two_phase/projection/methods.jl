"""Interface for selecting a two-velocity pressure projection."""
abstract type AbstractTwoVelocityProjection end

"""Pressure-projection policy for the conservative multidimensional ghost-fluid method.

The supported preconditioners are `:block` (the default), `:ilu`, and `nothing`;
unsupported choices are rejected at construction.
"""
struct CMDGFMProjection{P, O} <: AbstractTwoVelocityProjection
    preconditioner::P
    preconditioner_options::O

    function CMDGFMProjection(preconditioner, preconditioner_options)
        preconditioner in (:block, :ilu, nothing) || throw(
            ArgumentError("CMDGFM preconditioner must be :block, :ilu, or nothing"),
        )
        return new{typeof(preconditioner), typeof(preconditioner_options)}(
            preconditioner, preconditioner_options,
        )
    end
end

CMDGFMProjection(;
    preconditioner = :block, preconditioner_options = (;),
) = CMDGFMProjection(preconditioner, preconditioner_options)
