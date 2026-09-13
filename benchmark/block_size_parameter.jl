using BenchmarkTools

struct StaticLayout{NX,NY,H} end

Base.@noinline function dynamic_stencil!(output, input, nx, ny, halo)
    @inbounds for j in 1:ny, i in 1:nx
        ii, jj = i + halo, j + halo
        output[ii, jj] = 4input[ii, jj] - input[ii - 1, jj] -
                         input[ii + 1, jj] - input[ii, jj - 1] - input[ii, jj + 1]
    end
    return output
end

Base.@noinline function static_stencil!(output, input, ::StaticLayout{NX,NY,H}) where {NX,NY,H}
    @inbounds for j in 1:NY, i in 1:NX
        ii, jj = i + H, j + H
        output[ii, jj] = 4input[ii, jj] - input[ii - 1, jj] -
                         input[ii + 1, jj] - input[ii, jj - 1] - input[ii, jj + 1]
    end
    return output
end

println("Julia ", VERSION, ", threads=", Threads.nthreads())
println("CPU: ", Sys.cpu_info()[1].model)

for n in (8, 16, 32, 64)
    input = rand(n + 2, n + 2)
    output = similar(input)
    dynamic_trial = @benchmark dynamic_stencil!($output, $input, $n, $n, 1)
    layout = StaticLayout{n,n,1}()
    static_trial = @benchmark static_stencil!($output, $input, $layout)
    dynamic_estimate = median(dynamic_trial)
    static_estimate = median(static_trial)
    ratio = dynamic_estimate.time / static_estimate.time
    println("n=", n,
            ": dynamic=", dynamic_estimate.time, " ns, static=",
            static_estimate.time, " ns, dynamic/static=", round(ratio; digits=3))
end
