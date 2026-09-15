"""Supertype for locations at which discrete unknowns live."""
abstract type AbstractPlacement end

"""Supertype for cell-, face-, and vector-valued data attached to one mesh."""
abstract type AbstractMeshField{T} end

"""Cell-centred placement."""
struct Cell <: AbstractPlacement end

"""
    Face{Axis}()

Placement on faces normal to coordinate direction `Axis`.
"""
struct Face{Axis} <: AbstractPlacement
    function Face{Axis}() where {Axis}
        Axis in (1, 2) || throw(ArgumentError("the 2-D backend only supports axes 1 and 2"))
        new{Axis}()
    end
end
