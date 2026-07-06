The formulas chosen for the heuristic are designed to map the geometric properties of a point cloud directly to the discrete mathematical structures used by the PoissonRecon algorithm.
Since PoissonRecon converts your points into a continuous field using a hierarchical octree, the optimal parameters are entirely dependent on spatial scale and point sampling distribution. Here is the step-by-step reasoning behind each formula:
------------------------------
## 1. Depth Formula: $\text{depth} = \text{round}\left( \log_2\left( \frac{D}{\bar{d}_k} \right) \right) + C$
This formula solves the biggest issue in automated mesh reconstruction: matching the physical voxel resolution of the octree to the true resolution of your sensor data.

* The Ratio ($\frac{D}{\bar{d}_k}$): The bounding box diagonal ($D$) represents the maximum extent of your data. The average $k$-nearest neighbor distance ($\bar{d}_k$) represents the average space between adjacent points. Dividing them calculates how many point-sized steps are needed to cross the entire dataset along its longest dimension.
* The Logarithm ($\log_2$): An octree is a tree data structure where each node splits into $2^3 = 8$ children. This means the number of voxels along any axis at a given depth $d$ is exactly $2^d$. To find what depth $d$ gives us the target number of voxels, we must take the base-2 logarithm ($\log_2$) of our ratio.
* The Buffer ($C = +1$ or $+2$): According to Nyquist-Shannon sampling theorems, to accurately reconstruct a signal (the surface) defined by samples (the points), your sampling grid (the voxels) must be finer than the spacing of the samples. Adding $+1$ or $+2$ ensures the octree voxels are slightly smaller than $\bar{d}_k$. This allows the solver to capture the high-frequency surface features without introducing aliasing or voxel grid artifacts.

------------------------------
## 2. Point Weight Formula: $\text{pointWeight} = \max\left(2.0, \;\; 12.0 - (\text{depth} - 8) \times 1.5\right)$
In PoissonRecon, the --pointWeight acts as a regularization parameter. It balances the Screened Poisson equation between solving the gradient field (smoothing) and interpolating the point locations (data fidelity).

* Why an inverse relationship? As the depth of your tree increases, the size of individual leaf nodes shrinks exponentially. Because your nodes are tiny, the solver naturally becomes highly localized. If you keep a high pointWeight at a very high depth (like depth 11 or 12), the matrix system becomes rigid, overfits to tiny local variations, and can create micro-jagged edges.
* How the linear decay works:
* At a baseline depth of 8, the formula yields $12.0 - (0) = 12.0$. This strong weight forces the surface to strictly bind to the points because a lower-resolution grid needs stronger constraints to not over-smooth the details.
   * At a high depth of 12, the formula yields $12.0 - (4 \times 1.5) = 6.0$. At this resolution, the grid itself is dense enough to trap the details, so lowering the weight allows the Poisson field to stitch the micro-gaps smoothly.
* The Floor ($\max(2.0, \dots)$): This acts as a safety guardrail. If the weight drops too close to 0, the screening constraint is lost, and the algorithm turns into a standard, non-screened Poisson reconstruction, causing the mesh to shrink and lose its boundaries.

------------------------------
## 3. Samples Per Node: Variance-Driven $1.0$ vs $1.5$
This parameter sets the minimum weight of points required inside an octree node before it stops subdividing. It essentially acts as a low-pass filter for data density.

* Why variance matters: In an ideal world with completely uniform point clouds, every octree node at the target depth would contain exactly the same number of points. In reality, sensor data has variable density (e.g., a LiDAR scanner collects millions of points close to the sensor, but very few far away).
* The Logic:
* When density variance is low ($\leq 0.5$), the cloud is uniform. We set --samplesPerNode 1.0 so the octree can freely subdivide right down to individual precise points without being restricted.
   * When density variance is high ($> 0.5$), the cloud has massive gaps or sudden dense clusters. Setting --samplesPerNode 1.5 prevents the octree from over-subdividing in ultra-dense clusters (which wastes RAM) and helps stabilize the boundary transitions between the dense and sparse areas of your cloud.

Would you like to explore how to dynamically segment your point clouds if they contain both massive regional spaces (like a room) and highly detailed objects (like furniture) at the same time, or does this global heuristic cover your typical variation?

