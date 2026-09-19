@testset "Visualization frames retain PLIC interface and solid overlays" begin
    mesh = BlockTreeMesh(block_size = (8, 8))
    geometry = GeometryField(
        mesh, (x, y) -> hypot(x - 0.5, y - 0.5) - 0.2,
    )
    interface = InterfaceField(
        mesh, (x, y) -> y - 0.6; geometry,
    )
    registry = VisualizationFieldRegistry()
    register_default_fields!(registry, :volume_fraction, interface)
    @test registry.interfaces == [interface]
    @test registry.geometries == [geometry]
    frame = TwoPhaseAMR._capture_frame(registry, 0.0, 0)
    @test !isempty(frame.primitives[interface])
    @test !isempty(frame.primitives[geometry].triangles)
end
