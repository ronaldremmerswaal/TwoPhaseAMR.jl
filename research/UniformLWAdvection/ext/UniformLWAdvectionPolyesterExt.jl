module UniformLWAdvectionPolyesterExt

using UniformLWAdvection
using Polyester

# Explicit per=thread is essential: Polyester otherwise defaults to physical
# cores, so launching 32 Julia workers would not establish 32 participants.
function UniformLWAdvection.foreach_tile!(f, ::UniformLWAdvection.PolyesterBackend,
                                         tiles, chunks)
    Polyester.@batch per=thread minbatch=1 for t in eachindex(tiles)
        f(t)
    end
    nothing
end

end
