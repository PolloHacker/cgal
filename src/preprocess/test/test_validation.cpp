#include "doctest.h"
#include "validation.h"

using namespace mesh_reconstruction;

TEST_CASE("Testing validation and spacing calculations") {
    Point_set points;
    points.add_normal_map();
    
    Point_set::Index idx1 = *(points.insert(Point(1.0, 2.0, 3.0)));
    Point_set::Index idx2 = *(points.insert(Point(4.0, 5.0, 6.0)));
    Point_set::Index idx3 = *(points.insert(Point(7.0, 8.0, 9.0)));
    
    points.normal(idx1) = Vector(0.0, 1.0, 0.0);
    points.normal(idx2) = Vector(0.0, 1.0, 0.0);
    points.normal(idx3) = Vector(0.0, 1.0, 0.0);
    
    SUBCASE("Basic validation") {
        bool ok = validate_point_set(points, "test context", true);
        CHECK(ok);
    }
    
    SUBCASE("Average spacing calculation") {
        double spacing = compute_average_spacing(points, 2);
        CHECK(spacing > 0.0);
        bool spacing_ok = validate_average_spacing(spacing, "test spacing");
        CHECK(spacing_ok);
    }
}
