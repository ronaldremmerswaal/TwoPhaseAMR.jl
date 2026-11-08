# Symbolic helpers shared by the audits of Chapter 6 (plain Julia + Symbolics.jl; run under --project=@claude-symbolics).
#
# Not a check by itself.  Include with
#     isdefined(Main, :MomentumSymbolic) || include(joinpath(@__DIR__, "..", "..", "_shared", "momentum_symbolic.jl"))
#     using .MomentumSymbolic
#
# A local patch around one unknown f of axis k: its two cells c^-, c^+ with arbitrary side partitions, the transverse faces of
# both cells and the cell across each of them.  The dual faces at f are built from the printed rules (N) and (T) of
# docs/techref/chapters/06_momentum_transport.tex (eq:momentum:weights), the dual divergence at f is accumulated with the
# orientations sigma_{f,g} = +1 (tail) / -1 (head), and the printed right-hand side of eq:momentum:identity is built next to it.

module MomentumSymbolic

using Symbolics

export expanded, iszero_sym, EMPTY, build_patch, frac_of, axial_faces, divergence_at_f, printed_rhs, predicted_residuals,
    all_fluxes, neighbours_of, events_at_f

expanded(e) = Symbolics.simplify(Symbolics.expand(e))
iszero_sym(e) = isequal(expanded(e), 0)

const FREE = Ref(0)
fresh(prefix) = Symbolics.variable(Symbol(prefix, FREE[] += 1))

const EMPTY = (; faces = Symbol[], fracs = Any[])

"""
A random patch.  `sides[(cell, s)]` = (faces, fractions) on side `s` (-1 low, +1 high) of an axial cell; `orient` = o_{c,face};
`transverse` = list of (name, minus cell, plus cell); `trans_of[cell]` = list of (name, o_{c,h}).  `constrained` makes every
side partition sum to one (last fraction = 1 - the others); otherwise the fractions are free symbols.
"""
function build_patch(rng; constrained)
    sides = Dict{Tuple{Symbol, Int}, typeof(EMPTY)}()
    orient = Dict{Tuple{Symbol, Symbol}, Int}()
    transverse = Tuple{Symbol, Symbol, Symbol}[]
    trans_of = Dict{Symbol, Vector{Tuple{Symbol, Int}}}()
    nface = Ref(0)
    newface(prefix) = Symbol(prefix, nface[] += 1)
    function add_side!(cell, s, faces)
        fr = Any[fresh("w") for _ in faces]
        constrained && (fr[end] = 1 - sum(fr[1:(end - 1)]; init = 0))
        sides[(cell, s)] = (; faces, fracs = fr)
        for face in faces
            orient[(cell, face)] = s
        end
        return
    end
    segs() = rand(rng, 1:3)
    # the two cells of f; f is the first segment on its side
    add_side!(:cm, -1, [newface("L") for _ in 1:segs()])
    add_side!(:cm, +1, vcat(:f, [newface("H") for _ in 1:(segs() - 1)]))
    add_side!(:cp, -1, vcat(:f, [newface("J") for _ in 1:(segs() - 1)]))
    add_side!(:cp, +1, [newface("R") for _ in 1:segs()])
    # transverse faces of cm and cp, and the cell across each
    for cell in (:cm, :cp)
        trans_of[cell] = Tuple{Symbol, Int}[]
        for (kind, o) in (("T", +1), ("B", -1)), _ in 1:rand(rng, 1:2)
            h = newface(string(kind, cell))
            u = Symbol("u", h)
            add_side!(u, -1, [newface("N") for _ in 1:segs()])
            add_side!(u, +1, [newface("N") for _ in 1:segs()])
            push!(transverse, o == +1 ? (h, cell, u) : (h, u, cell))
            push!(trans_of[cell], (h, o))
        end
    end
    return (; sides, orient, transverse, trans_of)
end

frac_of(patch, cell, s, face) = begin
    sd = patch.sides[(cell, s)]
    sd.fracs[findfirst(==(face), sd.faces)]
end
axial_faces(patch, cell) = vcat(patch.sides[(cell, -1)].faces, patch.sides[(cell, +1)].faces)

"""Dual-face events of the printed rules for every flux that can reach f; returns the axial and transverse dual divergence at f."""
function divergence_at_f(patch, Phi)
    (; sides, orient, transverse) = patch
    div_ax = 0
    div_tr = 0
    # rule (N): flux face fh of cell c, other faces f' of c: dual face (f', fh), weight (1/2) o_{c,fh} varpi_{c,f'}
    for cell in (:cm, :cp), fh in axial_faces(patch, cell), s in (-1, 1)
        for (fp, w) in zip(sides[(cell, s)].faces, sides[(cell, s)].fracs)
            fp == fh && continue
            value = orient[(cell, fh)] * w / 2 * Phi[fh]
            fp == :f && (div_ax += value)          # f is the tail
            fh == :f && (div_ax -= value)          # f is the head
        end
    end
    # rule (T): transverse face h with cells (m, p): dual face (f', f''), f' on side s of m, f'' on side s of p,
    # weight varpi varpi' / 2
    for (h, m, p) in transverse, s in (-1, 1)
        for (fa, a) in zip(sides[(m, s)].faces, sides[(m, s)].fracs), (fb, b) in zip(sides[(p, s)].faces, sides[(p, s)].fracs)
            value = a * b / 2 * Phi[h]
            fa == :f && (div_tr += value)
            fb == :f && (div_tr -= value)
        end
    end
    return div_ax, div_tr
end

"""
The partial volumes of the dual faces at f: `(kind, sigma, flux, magnitude, sign)` with `kind` `:ax` / `:tr`, `sigma` = sigma_{f,g}
(+1 tail, -1 head), the primal face `flux`, the positive weight magnitude (a symbolic expression in the fractions) and the sign
of the weight (the orientation of rule (N); +1 for rule (T)).  The partial volume leaves the control volume of f iff
`sigma * sign * sign(Phi_flux) > 0`.
"""
function events_at_f(patch)
    (; sides, orient, transverse) = patch
    events = Tuple{Symbol, Int, Symbol, Any, Int}[]
    for cell in (:cm, :cp), fh in axial_faces(patch, cell), s in (-1, 1)
        for (fp, w) in zip(sides[(cell, s)].faces, sides[(cell, s)].fracs)
            fp == fh && continue
            fp == :f && push!(events, (:ax, +1, fh, w / 2, orient[(cell, fh)]))
            fh == :f && push!(events, (:ax, -1, fh, w / 2, orient[(cell, fh)]))
        end
    end
    for (h, m, p) in transverse, s in (-1, 1)
        for (fa, a) in zip(sides[(m, s)].faces, sides[(m, s)].fracs), (fb, b) in zip(sides[(p, s)].faces, sides[(p, s)].fracs)
            fa == :f && push!(events, (:tr, +1, h, a * b / 2, 1))
            fb == :f && push!(events, (:tr, -1, h, a * b / 2, 1))
        end
    end
    return events
end

"""The printed right-hand side of eq:momentum:identity at f for t = k (axial) and t != k (transverse)."""
function printed_rhs(patch, Phi)
    (; orient, trans_of) = patch
    rhs_ax = 0
    rhs_tr = 0
    for cell in (:cm, :cp)
        s = orient[(cell, :f)]
        varpi = frac_of(patch, cell, s, :f)
        for fh in axial_faces(patch, cell)
            rhs_ax += varpi / 2 * orient[(cell, fh)] * Phi[fh]
        end
        for (h, o) in trans_of[cell]
            rhs_tr += varpi / 2 * o * Phi[h]
        end
    end
    return rhs_ax, rhs_tr
end

"""Closed-form residuals S1, S2 (valid for arbitrary side fractions)."""
function predicted_residuals(patch, Phi)
    (; sides, orient, transverse, trans_of) = patch
    S(cell) = sum(sum(sides[(cell, s)].fracs; init = 0) for s in (-1, 1))
    res_ax = -Phi[:f] * sum(orient[(cell, :f)] * S(cell) for cell in (:cm, :cp)) / 2
    res_tr = 0
    for cell in (:cm, :cp)
        s = orient[(cell, :f)]
        varpi = frac_of(patch, cell, s, :f)
        for (h, o) in trans_of[cell]
            (_, m, p) = transverse[findfirst(t -> t[1] == h, transverse)]
            other = m == cell ? p : m
            res_tr += varpi / 2 * o * Phi[h] * (sum(sides[(other, s)].fracs; init = 0) - 1)
        end
    end
    return res_ax, res_tr
end

function all_fluxes(patch)
    names = Set{Symbol}()
    for ((cell, s), sd) in patch.sides, face in sd.faces
        push!(names, face)
    end
    for (h, _, _) in patch.transverse
        push!(names, h)
    end
    return Dict(name => Symbolics.variable(Symbol("Phi_", name)) for name in names)
end

"""Neighbour cells across the transverse faces of `cell`, with h and o_{cell,h}."""
neighbours_of(patch, cell) = [
    (m == cell ? p : m, h, o) for (h, m, p) in patch.transverse for (hh, o) in patch.trans_of[cell] if hh == h
]


end # module MomentumSymbolic
