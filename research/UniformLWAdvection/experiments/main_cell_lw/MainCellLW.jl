"""Research-only loader for the uniform `CellField` LW comparison.

The prototype uses private mesh schedules, so it is evaluated in the
TwoPhaseAMR module after that package is loaded. It does not change package
source or exports. Load once per Julia process before constructing a workspace.
"""
module MainCellLW

import TwoPhaseAMR

if !isdefined(TwoPhaseAMR, :UniformCellLWWorkspace)
    Base.include(TwoPhaseAMR, joinpath(@__DIR__, "uniform_cell_lw.jl"))
end

const UniformCellLWWorkspace = TwoPhaseAMR.UniformCellLWWorkspace
const set_uniform_lw_velocity! = TwoPhaseAMR.set_uniform_lw_velocity!
const track_uniform_lw_workers! = TwoPhaseAMR.track_uniform_lw_workers!
const active_uniform_lw_workers = TwoPhaseAMR.active_uniform_lw_workers
const active_uniform_lw_halo_workers = TwoPhaseAMR.active_uniform_lw_halo_workers

export UniformCellLWWorkspace, set_uniform_lw_velocity!
export track_uniform_lw_workers!, active_uniform_lw_workers
export active_uniform_lw_halo_workers

end
