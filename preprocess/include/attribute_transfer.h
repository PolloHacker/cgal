#pragma once

#include "mesh_reconstruction.h"

namespace mesh_reconstruction {

/**
 * \brief Transfers attributes (normals, colors, intensity) from a source point set
 *        to a target point set using 1-NN lookup.
 * \details Dynamically checks which properties are registered on the source point set,
 *          registers matching properties on the target, builds a KD-tree on the source,
 *          and uses a 1-NN orthogonal query to map values.
 * \param source The source point set containing the original attributes.
 * \param target The target point set to populate with attributes (should already have points inserted).
 * \return true if attributes are successfully transferred, false otherwise.
 */
bool transfer_attributes(const Point_set &source, Point_set &target);

} // namespace mesh_reconstruction
