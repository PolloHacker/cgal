#ifndef VALIDATION_H
#define VALIDATION_H

#include "mesh_reconstruction.h"

#include <CGAL/compute_average_spacing.h>

#include <vector>

using Pwn = mesh_reconstruction::Pwn;

/** \brief Validates that a point set is usable for reconstruction stages. */
bool validate_point_set(const std::vector<Pwn> &points,
                        const char *context,
                        const bool require_oriented_normals);

/** \brief Computes average spacing using k = neighbors (defaults to 6) */
double compute_average_spacing(const std::vector<Pwn> &points, const int neighbors);

/** \brief Computes average spacing using k = min(neighbors, max_neighbors).  */
double compute_average_spacing(const std::vector<Pwn> &points, const int neighbors, const unsigned int max);

/** \brief Validates spacing values used by Poisson reconstruction. */
bool validate_average_spacing(const double average_spacing, const char *context);

#endif