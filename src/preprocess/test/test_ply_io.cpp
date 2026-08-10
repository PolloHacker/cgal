#include "doctest.h"
#include "ply_io.h"
#include <fstream>
#include <string>
#include <cstdio>

using namespace mesh_reconstruction;

TEST_CASE("Testing ply_io loading and saving") {
    std::string temp_ply = "temp_test_ply_io.ply";
    std::ofstream out(temp_ply);
    REQUIRE(out.is_open());
    
    out << "ply\n"
        << "format ascii 1.0\n"
        << "element vertex 4\n"
        << "property double x\n"
        << "property double y\n"
        << "property double z\n"
        << "property double nx\n"
        << "property double ny\n"
        << "property double nz\n"
        << "property uchar red\n"
        << "property uchar green\n"
        << "property uchar blue\n"
        << "end_header\n"
        << "0.0 0.0 0.0 0.0 0.0 1.0 255 0 0\n"
        << "0.1 0.1 0.1 0.0 0.0 1.0 255 0 0\n"
        << "5.0 5.0 5.0 0.0 1.0 0.0 0 255 0\n"
        << "10.0 10.0 10.0 1.0 0.0 0.0 0 0 255\n";
    out.close();
    
    SUBCASE("Load without subsampling") {
        Point_set points;
        bool load_ok = load_ply(temp_ply, points, false);
        CHECK(load_ok);
        CHECK(points.size() == 4);
        CHECK(points.has_normal_map());
        
        auto red_prop = points.property_map<unsigned char>("red");
        CHECK(red_prop.second);
        
        // Write to another file to verify writing
        std::string temp_out_ply = "temp_test_ply_io_out.ply";
        bool write_ok = write_ply(temp_out_ply, points, true);
        CHECK(write_ok);
        
        Point_set read_points;
        bool reload_ok = load_ply(temp_out_ply, read_points, false);
        CHECK(reload_ok);
        CHECK(read_points.size() == 4);
        
        std::remove(temp_out_ply.c_str());
    }
    
    SUBCASE("Load with voxel subsampling (min_distance = 1.0)") {
        Point_set points;
        bool load_ok = load_ply(temp_ply, points, true, 1.0);
        CHECK(load_ok);
        CHECK(points.size() == 3);
    }
    
    std::remove(temp_ply.c_str());
}
