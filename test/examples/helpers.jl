"""Load one standalone example without leaking its `run_example` method."""
function load_example(name)
    example = Module(gensym(:Example))
    Base.include(
        example,
        joinpath(@__DIR__, "..", "..", "examples", string(name, ".jl")),
    )
    return example
end

function call_example(example, name; options...)
    entrypoint = Base.invokelatest(getproperty, example, name)
    return Base.invokelatest(entrypoint; options...)
end
