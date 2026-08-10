#include "doctest.h"
#include "mesh_reconstruction.h"

using namespace mesh_reconstruction;

TEST_CASE("Testing mesh normalization for skeletonization") {
    Triangle_mesh mesh;
    
    // Add 8 vertices of a cube
    auto v0 = mesh.add_vertex(Point(0.0, 0.0, 0.0));
    auto v1 = mesh.add_vertex(Point(1.0, 0.0, 0.0));
    auto v2 = mesh.add_vertex(Point(1.0, 1.0, 0.0));
    auto v3 = mesh.add_vertex(Point(0.0, 1.0, 0.0));
    auto v4 = mesh.add_vertex(Point(0.0, 0.0, 1.0));
    auto v5 = mesh.add_vertex(Point(1.0, 0.0, 1.0));
    auto v6 = mesh.add_vertex(Point(1.0, 1.0, 1.0));
    auto v7 = mesh.add_vertex(Point(0.0, 1.0, 1.0));
    
    // Add faces, leaving the top face (v4, v5, v6, v7) open (creating a hole)
    // Bottom face (triangulated)
    mesh.add_face(v3, v2, v1);
    mesh.add_face(v3, v1, v0);
    // Front face (triangulated)
    mesh.add_face(v0, v1, v5);
    mesh.add_face(v0, v5, v4);
    // Right face (triangulated)
    mesh.add_face(v1, v2, v6);
    mesh.add_face(v1, v6, v5);
    // Back face (triangulated)
    mesh.add_face(v2, v3, v7);
    mesh.add_face(v2, v7, v6);
    // Left face (triangulated)
    mesh.add_face(v3, v0, v4);
    mesh.add_face(v3, v4, v7);
    
    // Verify it is NOT closed before normalization
    REQUIRE(!CGAL::is_closed(mesh));
    
    // Run normalization (which should extract boundary cycles, fill the hole, and stitch)
    bool ok = normalize_mesh_for_skeletonization(mesh, true);
    CHECK(ok);
    
    // Verify the mesh is now closed and triangulated
    CHECK(CGAL::is_closed(mesh));
    CHECK(CGAL::is_triangle_mesh(mesh));
}
