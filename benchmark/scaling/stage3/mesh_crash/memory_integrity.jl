#!/usr/bin/env julia

# One bounded, dependency-free buffer fill/readback per pattern. This is a
# diagnostic of this process only, not a hardware test or a performance study.
using SHA
using TOML

const MAX_BYTES = 1 << 30
const DEFAULT_BYTES = MAX_BYTES
const PATTERNS = (UInt64(1), UInt64(0xaaaaaaaaaaaaaaaa))

function usage()
    println("Usage: julia --startup-file=no --check-bounds=yes memory_integrity.jl --output NEW_DIR [--bytes N]")
    println("N must be a positive multiple of 8, at most 1073741824 (default: 1073741824).")
end

function arguments(args)
    "--help" in args && (usage(); return nothing)
    output = nothing
    bytes = DEFAULT_BYTES
    seen = Set{String}()
    index = 1
    while index <= length(args)
        flag = args[index]
        flag in ("--output", "--bytes") || error("unknown argument: $flag")
        flag in seen && error("duplicate argument: $flag")
        push!(seen, flag)
        index < length(args) || error("missing value for $flag")
        value = args[index + 1]
        if flag == "--output"
            isempty(value) && error("--output needs a nonempty path")
            output = abspath(value)
        else
            bytes = parse(Int, value)
        end
        index += 2
    end
    output === nothing && error("--output is required")
    0 < bytes <= MAX_BYTES && bytes % sizeof(UInt64) == 0 ||
        error("--bytes must be a positive multiple of 8 at most $MAX_BYTES")
    ispath(output) && error("refusing existing output path: $output")
    return (; output, bytes)
end

hex64(value::UInt64) = "0x" * string(value; base=16, pad=16)

function rss_kib()
    isfile("/proc/self/status") || return -1
    for line in eachline("/proc/self/status")
        if startswith(line, "VmRSS:")
            return parse(Int, split(line)[2])
        end
    end
    return -1
end

function event!(directory, phase; values=Dict{String,Any}())
    payload = Dict{String,Any}(
        "phase" => phase,
        "unix_seconds" => time(),
        "monotonic_ns" => time_ns(),
        "rss_kib" => rss_kib(),
    )
    merge!(payload, values)
    open(joinpath(directory, "events.toml"), "a") do io
        TOML.print(io, Dict("events" => [payload]))
        flush(io)
        ccall(:fsync, Cint, (Cint,), Base.fd(io)) == 0 ||
            error("event fsync failed")
    end
    return nothing
end

function write_receipt(directory, receipt)
    open(joinpath(directory, "receipt.toml"), "w") do io
        TOML.print(io, receipt)
        flush(io)
        ccall(:fsync, Cint, (Cint,), Base.fd(io)) == 0 ||
            error("receipt fsync failed")
    end
end

@noinline function fill_buffer!(buffer::Vector{UInt64}, pattern::UInt64)
    for index in eachindex(buffer)
        buffer[index] = pattern
    end
    return nothing
end

@noinline function scan_buffer(buffer::Vector{UInt64}, expected::UInt64)
    mismatch_count = 0
    first_mismatches = Dict{String,Any}[]
    checksum = UInt64(0xcbf29ce484222325)
    value_sum = zero(UInt64)
    for index in eachindex(buffer)
        actual = buffer[index]
        checksum = (checksum ⊻ actual ⊻ UInt64(index)) * UInt64(0x100000001b3)
        value_sum += actual
        if actual != expected
            mismatch_count += 1
            if length(first_mismatches) < 10
                push!(first_mismatches, Dict{String,Any}(
                    "index" => index,
                    "actual_hex" => hex64(actual),
                    "expected_hex" => hex64(expected),
                    "xor_hex" => hex64(actual ⊻ expected),
                ))
            end
        end
    end
    return Dict{String,Any}(
        "mismatch_count" => mismatch_count,
        "first_mismatches" => first_mismatches,
        "checksum_hex" => hex64(checksum),
        "value_sum_hex" => hex64(value_sum),
    )
end

function main(args)
    parsed = arguments(args)
    parsed === nothing && return 0
    directory, bytes = parsed.output, parsed.bytes
    mkpath(dirname(directory))
    mkdir(directory)
    receipt = Dict{String,Any}(
        "schema" => 1,
        "script_sha256" => bytes2hex(sha256(read(@__FILE__))),
        "julia_version" => string(VERSION),
        "julia_executable" => joinpath(Sys.BINDIR, Base.julia_exename()),
        "active_project" => something(Base.active_project(), ""),
        "threads" => Threads.nthreads(),
        "bytes" => bytes,
        "words" => bytes ÷ sizeof(UInt64),
        "check_bounds_option" => string(Base.JLOptions().check_bounds),
        "started_unix_seconds" => time(),
    )
    event!(directory, "before_allocation";
           values=Dict("bytes" => bytes, "words" => receipt["words"]))
    buffer = Vector{UInt64}(undef, receipt["words"])
    event!(directory, "after_allocation")
    results = Dict{String,Any}[]
    for (pass, pattern) in enumerate(PATTERNS)
        label = hex64(pattern)
        event!(directory, "before_fill"; values=Dict("pass" => pass, "pattern" => label))
        fill_started = time_ns()
        fill_buffer!(buffer, pattern)
        fill_seconds = (time_ns() - fill_started) / 1e9
        event!(directory, "after_fill"; values=Dict("pass" => pass, "pattern" => label))
        scan_started = time_ns()
        result = scan_buffer(buffer, pattern)
        scan_seconds = (time_ns() - scan_started) / 1e9
        result["pass"] = pass
        result["pattern_hex"] = label
        result["fill_seconds"] = fill_seconds
        result["scan_seconds"] = scan_seconds
        result["rss_kib"] = rss_kib()
        push!(results, result)
        event!(directory, "after_scan"; values=Dict(
            "pass" => pass, "pattern" => label,
            "mismatch_count" => result["mismatch_count"],
            "checksum_hex" => result["checksum_hex"],
        ))
    end
    receipt["passes"] = results
    receipt["finished_unix_seconds"] = time()
    receipt["max_observed_rss_kib"] = maximum(result["rss_kib"] for result in results)
    receipt["all_values_match"] = all(result["mismatch_count"] == 0 for result in results)
    write_receipt(directory, receipt)
    return receipt["all_values_match"] ? 0 : 2
end

exit(main(ARGS))
