#include "attribute_transfer.h"

#include <iostream>
#include <vector>
#include <CGAL/Kd_tree.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Search_traits_adapter.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>

namespace mesh_reconstruction {

namespace {

// KD-Tree Adapter for Index Lookup
struct Point_set_property_map {
    typedef Point_set::Index key_type;
    typedef Point value_type;
    typedef const Point& reference;
    typedef boost::lvalue_property_map_tag category;

    const Point_set* ps;
    Point_set_property_map(const Point_set* ps = nullptr) : ps(ps) {}

    reference operator[](key_type idx) const {
        return ps->point(idx);
    }
    friend reference get(const Point_set_property_map& map, key_type idx) {
        return map.ps->point(idx);
    }
};

typedef CGAL::Search_traits_3<mesh_reconstruction::Kernel> Base_traits;
typedef CGAL::Search_traits_adapter<Point_set::Index, Point_set_property_map, Base_traits> Traits;
typedef CGAL::Distance_adapter<Point_set::Index, Point_set_property_map, CGAL::Euclidean_distance<Base_traits>> Distance;
typedef CGAL::Orthogonal_k_neighbor_search<Traits, Distance> Neighbor_search;
typedef Neighbor_search::Tree Search_tree;

} // namespace

bool transfer_attributes(const Point_set &source, Point_set &target) {
    if (source.empty()) {
        std::cerr << "Warning: source point set is empty. No attributes to transfer.\n";
        return true;
    }
    
    // Build KD-tree on original source points
    Point_set_property_map ppmap(&source);
    Traits traits(ppmap);
    Distance distance(ppmap);
    Search_tree search_tree(source.begin(), source.end(), Search_tree::Splitter(), traits);
    
    // Identify available attributes in source and register/retrieve in target
    const bool has_normals = source.has_normal_map();
    if (has_normals && !target.has_normal_map()) {
        target.add_normal_map();
    }
    
    auto red_prop = source.property_map<unsigned char>("red");
    auto green_prop = source.property_map<unsigned char>("green");
    auto blue_prop = source.property_map<unsigned char>("blue");
    
    const bool has_red = red_prop.second;
    const bool has_green = green_prop.second;
    const bool has_blue = blue_prop.second;
    
    Point_set::Property_map<unsigned char> s_red_map, s_green_map, s_blue_map;
    Point_set::Property_map<unsigned char> t_red_map, t_green_map, t_blue_map;
    if (has_red) {
        s_red_map = red_prop.first;
        t_red_map = target.add_property_map<unsigned char>("red", 0).first;
    }
    if (has_green) {
        s_green_map = green_prop.first;
        t_green_map = target.add_property_map<unsigned char>("green", 0).first;
    }
    if (has_blue) {
        s_blue_map = blue_prop.first;
        t_blue_map = target.add_property_map<unsigned char>("blue", 0).first;
    }
    
    auto intensity_float_prop = source.property_map<float>("intensity");
    auto intensity_double_prop = source.property_map<double>("intensity");
    const bool has_intensity_float = intensity_float_prop.second;
    const bool has_intensity_double = intensity_double_prop.second;
    
    Point_set::Property_map<float> t_intensity_float_map;
    Point_set::Property_map<double> t_intensity_double_map;
    if (has_intensity_float) {
        t_intensity_float_map = target.add_property_map<float>("intensity", 0.0f).first;
    }
    if (has_intensity_double) {
        t_intensity_double_map = target.add_property_map<double>("intensity", 0.0).first;
    }
    
    // Perform 1-NN query for each target point to pull original attributes
    for (const auto& idx : target) {
        const Point &p = target.point(idx);
        Neighbor_search search(search_tree, p, 1, 0.0, true, distance);
        if (search.begin() != search.end()) {
            auto nearest_source_idx = search.begin()->first;
            
            if (has_normals) {
                target.normal(idx) = source.normal(nearest_source_idx);
            }
            if (has_red) {
                t_red_map[idx] = s_red_map[nearest_source_idx];
            }
            if (has_green) {
                t_green_map[idx] = s_green_map[nearest_source_idx];
            }
            if (has_blue) {
                t_blue_map[idx] = s_blue_map[nearest_source_idx];
            }
            if (has_intensity_float) {
                t_intensity_float_map[idx] = intensity_float_prop.first[nearest_source_idx];
            }
            if (has_intensity_double) {
                t_intensity_double_map[idx] = intensity_double_prop.first[nearest_source_idx];
            }
        }
    }
    
    return true;
}

} // namespace mesh_reconstruction
