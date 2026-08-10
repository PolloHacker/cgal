#ifndef VALIDATION_H
#define VALIDATION_H

#include "mesh_reconstruction.h"

#include <CGAL/compute_average_spacing.h>

#include <vector>

using Point_set = mesh_reconstruction::Point_set;

/** \brief Validates that a point set is usable for reconstruction stages. */
bool validate_point_set(const Point_set &points,
                        const char *context,
                        const bool require_oriented_normals);

#endif