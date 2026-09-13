import TwoPhaseAMR as TP
using SparseArrays: nnz

function graded_mesh()
    return TP.BlockTreeMesh(
        block_size=(16, 16), max_level=3,
        refine=(key, origin, extent) ->
            key.level == 0 ||
            (key.level < 3 && origin[1] < 0.5 && origin[2] < 0.5),
    )
end

function measured_construction(constructor)
    constructor() # compile before measuring
    GC.gc()
    return @timed constructor()
end

function report_hierarchy(name, preconditioner, levels;
                          fine_stiffness_nonzeros=nothing)
    number_of_unknowns = size(first(levels).matrix, 1)
    fine_nonzeros = nnz(first(levels).matrix)
    grid_complexity = sum(size(level.matrix, 1) for level in levels) /
                      number_of_unknowns
    operator_complexity = sum(nnz(level.matrix) for level in levels) /
                          fine_nonzeros

    println("\n## $name")
    println("grid complexity: $(round(grid_complexity; digits=3))")
    println("operator complexity: $(round(operator_complexity; digits=3))")
    println("retained bytes: $(Base.summarysize(preconditioner))")
    println("| level | unknowns | A nnz | A nnz/row | K nnz | P nnz |")
    println("|---:|---:|---:|---:|---:|---:|")
    for (index, level) in pairs(levels)
        transfer_nonzeros = index == length(levels) ? 0 :
                            nnz(level.prolongation)
        # The compact stable stiffness is used only while constructing the
        # hierarchy and is not retained separately from the shifted matrices.
        stiffness_nonzeros = index == 1 &&
            !isnothing(fine_stiffness_nonzeros) ?
            fine_stiffness_nonzeros : nnz(level.matrix)
        entries_per_row = nnz(level.matrix) / size(level.matrix, 1)
        println("| $index | $(size(level.matrix, 1)) | " *
                "$(nnz(level.matrix)) | " *
                "$(round(entries_per_row; digits=2)) | " *
                "$stiffness_nonzeros | $transfer_nonzeros |")
    end
end

mesh = graded_mesh()
println("Mesh: $(TP.ncells(mesh)) cells, $(length(mesh.faces)) owned faces")

pressure_operator = TP.PoissonOperator(mesh)
pressure_measurement = measured_construction(
    () -> TP.GeometricMultigridPreconditioner(
        pressure_operator; coarse_size=32))
pressure = pressure_measurement.value
println("\nPressure hierarchy construction")
println("cumulative allocated bytes: $(pressure_measurement.bytes)")
println("elapsed seconds: $(pressure_measurement.time)")
report_hierarchy("Pressure", pressure, pressure.levels)

diffusion = TP.StaggeredDiffusionOperator(
    mesh; viscosity=1.0, boundary=TP.FreeSlip())
maximum_rate = max(diffusion.x.maximum_row_sum,
                   diffusion.y.maximum_row_sum)
diffusion_operator = TP.StaggeredEllipticOperator(
    diffusion; diffusion=40 / maximum_rate)
diffusion_measurement = measured_construction(
    () -> TP.StaggeredGeometricMultigridPreconditioner(
        diffusion_operator; coarse_size=32))
diffusion_preconditioner = diffusion_measurement.value
println("\nDiffusion hierarchy construction")
println("cumulative allocated bytes: $(diffusion_measurement.bytes)")
println("elapsed seconds: $(diffusion_measurement.time)")
println("wrapper retained bytes: " *
        "$(Base.summarysize(diffusion_preconditioner))")
report_hierarchy(
    "Diffusion x", diffusion_preconditioner.x,
    diffusion_preconditioner.x.levels;
    fine_stiffness_nonzeros=nnz(diffusion.x.stiffness))
report_hierarchy(
    "Diffusion y", diffusion_preconditioner.y,
    diffusion_preconditioner.y.levels;
    fine_stiffness_nonzeros=nnz(diffusion.y.stiffness))
