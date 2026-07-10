#include "doctest.h"
#include "attribute_transfer.h"

using namespace mesh_reconstruction;

TEST_CASE("Testing attribute transfer") {
    Point_set source;
    Point_set::Index s1 = *(source.insert(Point(0.0, 0.0, 0.0)));
    Point_set::Index s2 = *(source.insert(Point(10.0, 10.0, 10.0)));
    
    source.add_normal_map();
    source.normal(s1) = Vector(0.0, 0.0, 1.0);
    source.normal(s2) = Vector(1.0, 0.0, 0.0);
    
    auto s_red_map = source.add_property_map<unsigned char>("red", 0).first;
    s_red_map[s1] = 255;
    s_red_map[s2] = 0;
    
    Point_set target;
    Point_set::Index t1 = *(target.insert(Point(0.05, 0.05, 0.05)));
    Point_set::Index t2 = *(target.insert(Point(9.95, 9.95, 9.95)));
    
    bool transfer_ok = transfer_attributes(source, target);
    CHECK(transfer_ok);
    
    CHECK(target.has_normal_map());
    CHECK(target.normal(t1) == Vector(0.0, 0.0, 1.0));
    CHECK(target.normal(t2) == Vector(1.0, 0.0, 0.0));
    
    auto t_red_map = target.property_map<unsigned char>("red");
    CHECK(t_red_map.second);
    CHECK(t_red_map.first[t1] == 255);
    CHECK(t_red_map.first[t2] == 0);
}
