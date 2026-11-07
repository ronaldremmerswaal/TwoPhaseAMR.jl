using Test
using Aqua
using TwoPhaseAMR

@testset "Aqua" begin
    Aqua.test_all(TwoPhaseAMR)
    if isdefined(Aqua, :test_undocumented_names)
        Aqua.test_undocumented_names(TwoPhaseAMR)
    end
end
