function _mesh_topology_changed(workspace::AMRWorkspace)
    length(workspace.leaf_nodes) == length(workspace.old_blocks) || return true
    for (node, block) in zip(workspace.leaf_nodes, workspace.old_blocks)
        node.data.key == block.key || return true
    end
    return false
end

function _actual_adaptation_counts!(workspace::AMRWorkspace, step)
    old_keys = workspace.old_indices
    new_keys = workspace.new_indices
    empty!(old_keys)
    for (index, block) in pairs(workspace.old_blocks)
        old_keys[block.key] = index
    end
    empty!(new_keys)
    for (index, block) in pairs(workspace.mesh.blocks)
        new_keys[block.key] = index
    end
    refined_parents = workspace.refined_parents
    empty!(refined_parents)
    coarsened_parents = 0
    for block in workspace.mesh.blocks
        haskey(old_keys, block.key) && continue
        parent = block.key.level > 0 ? _parent_key(block.key) : nothing
        if !isnothing(parent) && haskey(old_keys, parent)
            push!(refined_parents, parent)
            workspace.mesh.adaptation.last_refined_step[block.key] = step
        elseif all(haskey(old_keys, child) for child in _child_keys(block.key))
            coarsened_parents += 1
        end
    end
    return length(refined_parents), coarsened_parents
end

function _adapt_with_labels!(
        workspace::AMRWorkspace,
        labels::AbstractVector{Symbol}; step, force = false
    )
    mesh = workspace.mesh
    state = mesh.adaptation
    step = Int(step)
    step >= 0 || throw(ArgumentError("adaptation step must be nonnegative"))
    if !force && !isnothing(state.last_update_step) &&
            step - state.last_update_step < state.options.update_interval
        return AMRStats(false, false, nblocks(mesh), nblocks(mesh), 0, 0)
    end
    state.last_update_step = step
    old_count = nblocks(mesh)
    _copy_geometry_snapshot!(workspace)
    _apply_refinement_actions!(workspace, labels, step)
    _collect_active_nodes!(workspace)
    if !_mesh_topology_changed(workspace)
        return AMRStats(true, false, old_count, old_count, 0, 0)
    end

    _prepare_adaptive_phase_velocities!(workspace)
    for buffer in workspace.field_buffers
        _stage_field!(buffer, workspace)
    end
    _build_face_transfer_geometry!(workspace)
    _rebuild_mesh_schedules!(workspace)
    for buffer in workspace.field_buffers
        if buffer isa CellAMRFieldBuffer
            _transfer_cell_field!(buffer, workspace)
        else
            _transfer_face_field!(buffer, workspace)
        end
    end
    for buffer in workspace.field_buffers
        buffer isa CellAMRFieldBuffer && _finish_prolongation!(buffer.prolongation)
    end
    _finish_adaptive_phase_velocities!(workspace)
    refined, coarsened = _actual_adaptation_counts!(workspace, step)
    return AMRStats(true, true, old_count, nblocks(mesh), refined, coarsened)
end

"""
    adapt!(workspace, indicators; step, time=0, force=false)
    adapt!(workspace, labels::Vector{Symbol}; step, force=false)

Adapt the workspace mesh in place and transfer every registered field.
Indicator proposals are combined with `:refine > :keep > :coarsen` priority.
Coarsening requires all four active siblings to request it, respects the mesh
cooldown and minimum level, and is followed by refinement-only 2:1 balancing.
The mesh's update interval is enforced unless `force=true`.
"""
function adapt!(
        workspace::AMRWorkspace, indicators;
        step, time = 0, force = false
    )
    for indicator in indicators
        indicator isa AbstractRefinementIndicator || throw(
            ArgumentError(
                "adapt! expects refinement indicators or a Symbol label vector"
            )
        )
        _register_indicator_field!(workspace, indicator)
    end
    labels = refinement_labels!(workspace, indicators; time)
    return _adapt_with_labels!(workspace, labels; step, force)
end

adapt!(
    workspace::AMRWorkspace, indicator::AbstractRefinementIndicator;
    step, time = 0, force = false
) = adapt!(workspace, (indicator,); step, time, force)

function adapt!(
        workspace::AMRWorkspace, labels::AbstractVector{Symbol};
        step, force = false
    )
    return _adapt_with_labels!(workspace, labels; step, force)
end
