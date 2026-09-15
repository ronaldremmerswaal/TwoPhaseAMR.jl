using BenchmarkTools
using TwoPhaseAMR

function adaptation_case()
    mesh = BlockTreeMesh(
        block_size=(16, 16), max_level=3,
        adaptation=MeshAdaptationOptions(coarsen_delay=0),
        refine=(key, origin, extent) -> key.level < 2)
    first = CellField(mesh)
    second = CellField(mesh)
    initialize!(first, (x, y) -> 1 + 0.2sinpi(4x) * cospi(2y))
    initialize!(second, (x, y) -> x + 2y)
    velocity = FaceVelocity(
        mesh,
        (x, y, time, axis) -> axis == 1 ?
            sinpi(2x) * cospi(2y) : -cospi(2x) * sinpi(2y))
    xface = FaceField{1}(mesh, (x, y) -> x - y)
    yface = FaceField{2}(mesh, (x, y) -> x + y)
    workspace = AMRWorkspace(
        mesh, first, second, velocity, xface, yface)
    labels = [isodd(block.key.i + block.key.j) ? :refine : :keep
              for block in mesh.blocks]
    return workspace, labels
end

function refined_adaptation_case()
    workspace, labels = adaptation_case()
    adapt!(workspace, labels; step=0, force=true)
    coarsen = [block.key.level == 3 ? :coarsen : :keep
               for block in workspace.mesh.blocks]
    return workspace, coarsen
end

function report(label, trial)
    estimate = BenchmarkTools.median(trial)
    println(rpad(label, 35),
            " time = ", BenchmarkTools.prettytime(estimate.time),
            ", memory = ", BenchmarkTools.prettymemory(estimate.memory),
            ", allocs = ", estimate.allocs)
end

halo = 1
parent = zeros(18, 18)
for j in axes(parent, 2), i in axes(parent, 1)
    parent[i, j] = sinpi(i / 18) * cospi(j / 18)
end
children = ntuple(_ -> similar(parent), 4)

prolong_trial = @benchmark prolong!($children, $parent, $halo)
restrict_trial = @benchmark restrict!($parent, $children, $halo)
refine_trial = @benchmark adapt!(workspace, labels; step=0, force=true) setup=(
    (workspace, labels) = adaptation_case()) samples=20 seconds=5 evals=1
coarsen_trial = @benchmark adapt!(workspace, labels; step=1, force=true) setup=(
    (workspace, labels) = refined_adaptation_case()) samples=20 seconds=5 evals=1

println("Adaptive refinement benchmark")
report("cell prolong!", prolong_trial)
report("cell restrict!", restrict_trial)
report("full refine + five fields", refine_trial)
report("full coarsen + five fields", coarsen_trial)
