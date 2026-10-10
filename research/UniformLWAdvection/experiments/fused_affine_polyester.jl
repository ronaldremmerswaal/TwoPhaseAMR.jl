# Included into UniformLWFusedAffineExperiment only when Polyester is visible.
function compute_tiles!(s::UniformLWAdvection.State,
                        w::FusedWorkspace{UniformLWAdvection.PolyesterBackend},
                        q0,out,rx,ry)
    if Threads.nthreads(:default) == 1 || length(q0) < w.threshold
        for c in eachindex(w.chunks)
            scratch = @inbounds w.scratch[c]
            for t in w.chunks[c]
                affine_tile!(s,w,scratch,w.tiles[t],q0,out,rx,ry)
            end
        end
    else
        Polyester.@batch per=thread minbatch=1 for c in eachindex(w.chunks)
            scratch = @inbounds w.scratch[c]
            for t in w.chunks[c]
                affine_tile!(s,w,scratch,w.tiles[t],q0,out,rx,ry)
            end
        end
    end
    nothing
end

function copyback!(s::UniformLWAdvection.State,
                   w::FusedWorkspace{UniformLWAdvection.PolyesterBackend},source)
    if Threads.nthreads(:default) == 1 || length(source) < w.threshold
        for c in eachindex(w.chunks), t in w.chunks[c]
            BaseFused.copy_tile!(s.q,source,w.tiles[t])
        end
    else
        Polyester.@batch per=thread minbatch=1 for c in eachindex(w.chunks)
            for t in w.chunks[c]
                BaseFused.copy_tile!(s.q,source,w.tiles[t])
            end
        end
    end
    nothing
end

function compute_tiles!(s::UniformLWAdvection.State,
                        w::FusedWorkspace{UniformLWAdvection.PolyesterBackend},
                        q0,out,rx,ry,probe::WorkerProbe,pass::Int)
    if Threads.nthreads(:default) == 1 || length(q0) < w.threshold
        for c in eachindex(w.chunks)
            @inbounds probe.compute[c,pass] = Threads.threadid()
            scratch = @inbounds w.scratch[c]
            for t in w.chunks[c]
                affine_tile!(s,w,scratch,w.tiles[t],q0,out,rx,ry)
            end
        end
    else
        Polyester.@batch per=thread minbatch=1 for c in eachindex(w.chunks)
            @inbounds probe.compute[c,pass] = Threads.threadid()
            scratch = @inbounds w.scratch[c]
            for t in w.chunks[c]
                affine_tile!(s,w,scratch,w.tiles[t],q0,out,rx,ry)
            end
        end
    end
    nothing
end

function copyback!(s::UniformLWAdvection.State,
                   w::FusedWorkspace{UniformLWAdvection.PolyesterBackend},
                   source,probe::WorkerProbe)
    if Threads.nthreads(:default) == 1 || length(source) < w.threshold
        for c in eachindex(w.chunks)
            @inbounds probe.copyback[c] = Threads.threadid()
            for t in w.chunks[c]
                BaseFused.copy_tile!(s.q,source,w.tiles[t])
            end
        end
    else
        Polyester.@batch per=thread minbatch=1 for c in eachindex(w.chunks)
            @inbounds probe.copyback[c] = Threads.threadid()
            for t in w.chunks[c]
                BaseFused.copy_tile!(s.q,source,w.tiles[t])
            end
        end
    end
    nothing
end
