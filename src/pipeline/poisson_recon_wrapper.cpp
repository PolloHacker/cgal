// SPDX-License-Identifier: MIT
#include "poisson_recon_wrapper.h"

// Disable unused variables warning in Kazhdan code
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wreorder"

#include "PreProcessor.h"
#include "Reconstructors.h"
#include "MyMiscellany.h"
#include "CmdLineParser.h"
#include "PPolynomial.h"
#include "FEMTree.h"
#include "Ply.h"
#include "VertexFactory.h"
#include "RegularGrid.h"
#include "DataStream.imp.h"

#pragma GCC diagnostic pop

#include <unordered_map>
#include <unordered_set>
#include <list>
#include <vector>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <cmath>
#include <iostream>

namespace {

using namespace PoissonRecon;

// Trimming helpers ported from SurfaceTrimmer.cpp
template< typename Index >
size_t BoostHash( Index i1 , Index i2 )
{
	size_t hash = (size_t)i1 + 0x9e3779b9;
	hash ^= (size_t)i2 + 0x9e3779b9 + (hash<<6) + (hash>>2);
	return hash;
}

template< typename Index >
struct EdgeKey
{
	Index key1 , key2;
	EdgeKey( Index k1=0 , Index k2=0 )
	{
		key1 = std::min< Index >( k1 , k2 );
		key2 = std::max< Index >( k1 , k2 );
	}
	bool operator == ( const EdgeKey &key ) const  { return key1==key.key1 && key2==key.key2; }
	struct Hasher{ size_t operator()( const EdgeKey &key ) const { return BoostHash(key.key1,key.key2); } };
};

template< typename Index >
struct HalfEdgeKey
{
	Index key1 , key2;
	HalfEdgeKey( Index k1=0 , Index k2=0 ) : key1(k1) , key2(k2) {}
	HalfEdgeKey opposite( void ) const { return HalfEdgeKey( key2 , key1 ); }
	bool operator == ( const HalfEdgeKey &key ) const  { return key1==key.key1 && key2==key.key2; }
	struct Hasher{ size_t operator()( const HalfEdgeKey &key ) const { return BoostHash(key.key1,key.key2); } };
};

template< typename Index >
struct ComponentGraph
{
	struct Node
	{
		double area;
		std::vector< Node * > neighbors;
		std::list< Index > polygonIndices;

		Node( void ) : area(0) {}

		void merge( void )
		{
			auto PopBack =[]( std::vector< Node * > &nodes , size_t idx )
			{
				nodes[idx] = nodes.back();
				nodes.pop_back();
			};

			if( !neighbors.size() ) return;

			// Remove the node from the neighbors of the neighbors
			for( unsigned int i=0 ; i<neighbors.size() ; i++ ) for( int j=(int)neighbors[i]->neighbors.size()-1 ; j>=0 ; j-- ) if( neighbors[i]->neighbors[j]==this )
				PopBack( neighbors[i]->neighbors , j );

			// Merge the node into its first neighbor
			Node *first = neighbors[0];
			first->area += area;
			first->polygonIndices.splice( first->polygonIndices.end() , polygonIndices );

			// Merge the remaining neighbors into the first neighbor
			for( unsigned int i=1 ; i<neighbors.size() ; i++ )
			{
				first->area += neighbors[i]->area;
				first->polygonIndices.splice( first->polygonIndices.end() , neighbors[i]->polygonIndices );
				for( unsigned int j=0 ; j<neighbors[i]->neighbors.size() ; j++ )
				{
					bool foundNeighbor = false;
					for( int k=(int)neighbors[i]->neighbors[j]->neighbors.size()-1 ; k>=0 ; k-- )
						if( neighbors[i]->neighbors[j]->neighbors[k]==neighbors[i] ) PopBack( neighbors[i]->neighbors[j]->neighbors , k );

					for( unsigned int k=0 ; k<first->neighbors.size() ; k++ ) foundNeighbor |= neighbors[i]->neighbors[j]==first->neighbors[k];
					if( !foundNeighbor )
					{
						first->neighbors.push_back( neighbors[i]->neighbors[j] );
						neighbors[i]->neighbors[j]->neighbors.push_back( first );
					}
				}

				neighbors[i]->area = 0;
				neighbors[i]->neighbors.clear();
			}
			// Clean up the node
			polygonIndices.clear();
			area = 0;
		}
	};
};

template< typename Index , typename Vertex, typename Real, unsigned int Dim >
Vertex InterpolateVertices( const Vertex& v1 , const Vertex& v2 , Real value )
{
	Real dx = ( v1.template get<1>()-value ) / ( v1.template get<1>()-v2.template get<1>() );
	Vertex v;
	v.template get<0>() = v1.template get<0>() * dx + v2.template get<0>() * ( (Real)1. - dx );
	v.template get<1>() = value;
	return v;
}

template< typename Real, unsigned int Dim, typename Index, class Vertex >
void SplitPolygon
(
	const std::vector< Index >& polygon ,
	std::vector< Vertex >& vertices ,
	std::vector< std::vector< Index > >* ltPolygons , std::vector< std::vector< Index > >* gtPolygons ,
	std::vector< bool >* ltFlags , std::vector< bool >* gtFlags ,
	std::unordered_map< EdgeKey< Index > , Index , typename EdgeKey< Index >::Hasher >& vertexTable,
	Real trimValue
)
{
	int sz = int( polygon.size() );
	std::vector< bool > gt( sz );
	int gtCount = 0;
	for( int j=0 ; j<sz ; j++ )
	{
		gt[j] = ( vertices[ polygon[j] ].template get<1>()>trimValue );
		if( gt[j] ) gtCount++;
	}
	if     ( gtCount==sz ){ if( gtPolygons ) gtPolygons->push_back( polygon ) ; if( gtFlags ) gtFlags->push_back( false ); }
	else if( gtCount==0  ){ if( ltPolygons ) ltPolygons->push_back( polygon ) ; if( ltFlags ) ltFlags->push_back( false ); }
	else
	{
		int start;
		for( start=0 ; start<sz ; start++ ) if( gt[start] && !gt[(start+sz-1)%sz] ) break;

		bool gtFlag = true;
		std::vector< Index > poly;

		// Add the initial vertex
		{
			int j1 = (start+int(sz)-1)%sz , j2 = start;
			Index v1 = polygon[j1] , v2 = polygon[j2] , vIdx;
			typename std::unordered_map< EdgeKey< Index > , Index , typename EdgeKey< Index >::Hasher >::iterator iter = vertexTable.find( EdgeKey< Index >(v1,v2) );
			if( iter==vertexTable.end() )
			{
				vertexTable[ EdgeKey< Index >(v1,v2) ] = vIdx = (Index)vertices.size();
				vertices.push_back( InterpolateVertices<Index, Vertex, Real, Dim>( vertices[v1] , vertices[v2] , trimValue ) );
			}
			else vIdx = iter->second;
			poly.push_back( vIdx );
		}

		for( int _j=0  ; _j<=sz ; _j++ )
		{
			int j1 = (_j+start+sz-1)%sz , j2 = (_j+start)%sz;
			Index v1 = polygon[j1] , v2 = polygon[j2];
			if( gt[j2]==gtFlag ) poly.push_back( v2 );
			else
			{
				Index vIdx;
				typename std::unordered_map< EdgeKey< Index > , Index , typename EdgeKey< Index >::Hasher >::iterator iter = vertexTable.find( EdgeKey< Index >(v1,v2) );
				if( iter==vertexTable.end() )
				{
					vertexTable[ EdgeKey< Index >(v1,v2) ] = vIdx = (Index)vertices.size();
					vertices.push_back( InterpolateVertices<Index, Vertex, Real, Dim>( vertices[v1] , vertices[v2] , trimValue ) );
				}
				else vIdx = iter->second;
				poly.push_back( vIdx );
				if( gtFlag ){ if( gtPolygons ) gtPolygons->push_back( poly ) ; if( ltFlags ) ltFlags->push_back( true ); }
				else        { if( ltPolygons ) ltPolygons->push_back( poly ) ; if( gtFlags ) gtFlags->push_back( true ); }
				poly.clear() , poly.push_back( vIdx ) , poly.push_back( v2 );
				gtFlag = !gtFlag;
			}
		}
	}
}

template< typename Index >
void SetConnectedComponents( const std::vector< std::vector< Index > >& polygons , std::vector< std::vector< Index > >& components )
{
	std::vector< Index > polygonRoots( polygons.size() );
	for( size_t i=0 ; i<polygons.size() ; i++ ) polygonRoots[i] = (Index)i;
	std::unordered_map< EdgeKey< Index > , Index , typename EdgeKey< Index >::Hasher > edgeTable;
	for( size_t i=0 ; i<polygons.size() ; i++ )
	{
		int sz = int( polygons[i].size() );
		for( int j=0 ; j<sz ; j++ )
		{
			int j1 = j , j2 = (j+1)%sz;
			Index v1 = polygons[i][j1] , v2 = polygons[i][j2];
			EdgeKey< Index > eKey = EdgeKey< Index >(v1,v2);
			typename std::unordered_map< EdgeKey< Index > , Index , typename EdgeKey< Index >::Hasher >::iterator iter = edgeTable.find(eKey);
			if( iter==edgeTable.end() ) edgeTable[ eKey ] = (Index)i;
			else
			{
				Index p = iter->second;
				while( polygonRoots[p]!=p )
				{
					Index temp = polygonRoots[p];
					polygonRoots[p] = (Index)i;
					p = temp;
				}
				polygonRoots[p] = (Index)i;
			}
		}
	}
	for( size_t i=0 ; i<polygonRoots.size() ; i++ )
	{
		Index p = (Index)i;
		while( polygonRoots[p]!=p ) p = polygonRoots[p];
		Index root = p;
		p = (Index)i;
		while( polygonRoots[p]!=p )
		{
			Index temp = polygonRoots[p];
			polygonRoots[p] = root;
			p = temp;
		}
	}
	int cCount = 0;
	std::unordered_map< Index , Index > vMap;
	for( Index i=0 ; i<(Index)polygonRoots.size() ; i++ ) if( polygonRoots[i]==i ) vMap[i] = cCount++;
	components.resize( cCount );
	for( Index i=0 ; i<(Index)polygonRoots.size() ; i++ ) components[ vMap[ polygonRoots[i] ] ].push_back(i);
}

template< class Real , unsigned int Dim , typename Index , class Vertex >
void Triangulate( const std::vector< Vertex >& vertices , const std::vector< std::vector< Index > >& polygons , std::vector< std::vector< Index > >& triangles )
{
	triangles.clear();
	for( size_t i=0 ; i<polygons.size() ; i++ )
		if( polygons[i].size()>3 )
		{
			std::vector< Point< Real , Dim > > _vertices( polygons[i].size() );
			for( int j=0 ; j<int( polygons[i].size() ) ; j++ ) _vertices[j] = vertices[ polygons[i][j] ].template get<0>();
			std::vector< TriangleIndex< Index > > _triangles = MinimalAreaTriangulation< Index , Real , Dim >( ( ConstPointer( Point< Real , Dim > ) )GetPointer( _vertices ) , _vertices.size() );

			// Add the triangles to the mesh
			size_t idx = triangles.size();
			triangles.resize( idx+_triangles.size() );
			for( int j=0 ; j<int(_triangles.size()) ; j++ )
			{
				triangles[idx+j].resize(3);
				for( int k=0 ; k<3 ; k++ ) triangles[idx+j][k] = polygons[i][ _triangles[j].idx[k] ];
			}
		}
		else if( polygons[i].size()==3 ) triangles.push_back( polygons[i] );
}

template< class Real , unsigned int Dim , typename Index , class Vertex >
double PolygonArea( const std::vector< Vertex >& vertices , const std::vector< Index >& polygon )
{
	auto Area =[]( Point< Real , Dim > v1 , Point< Real , Dim > v2 , Point< Real , Dim > v3 )
	{
		Point< Real , Dim > v[] = { v2-v1 , v3-v1 };
		XForm< Real , 2 > Mass;
		for( int i=0 ; i<2 ; i++ ) for( int j=0 ; j<2 ; j++ ) Mass(i,j) = Point< Real , Dim >::Dot( v[i] , v[j] );
		double det = Mass.determinant();
		if( det<0 ) return (Real)0;
		else return (Real)( sqrt( Mass.determinant() ) / 2. );
	};

	if( polygon.size()<3 ) return 0.;
	else if( polygon.size()==3 ) return Area( vertices[polygon[0]].template get<0>() , vertices[polygon[1]].template get<0>() , vertices[polygon[2]].template get<0>() );
	else
	{
		Point< Real , Dim > center;
		for( size_t i=0 ; i<polygon.size() ; i++ ) center += vertices[ polygon[i] ].template get<0>();
		center /= Real( polygon.size() );
		double area = 0;
		for( size_t i=0 ; i<polygon.size() ; i++ ) area += Area( center , vertices[ polygon[i] ].template get<0>() , vertices[ polygon[ (i+1)%polygon.size() ] ].template get<0>() );
		return area;
	}
}

template< typename Index , class Vertex >
void RemoveHangingVertices( std::vector< Vertex >& vertices , std::vector< std::vector< Index > >& polygons )
{
	std::unordered_map< Index, Index > vMap;
	std::vector< bool > vertexFlags( vertices.size() , false );
	for( size_t i=0 ; i<polygons.size() ; i++ ) for( size_t j=0 ; j<polygons[i].size() ; j++ ) vertexFlags[ polygons[i][j] ] = true;
	Index vCount = 0;
	for( Index i=0 ; i<(Index)vertices.size() ; i++ ) if( vertexFlags[i] ) vMap[i] = vCount++;
	for( size_t i=0 ; i<polygons.size() ; i++ ) for( size_t j=0 ; j<polygons[i].size() ; j++ ) polygons[i][j] = vMap[ polygons[i][j] ];

	std::vector< Vertex > _vertices( vCount );
	for( Index i=0 ; i<(Index)vertices.size() ; i++ ) if( vertexFlags[i] ) _vertices[ vMap[i] ] = vertices[i];
	vertices = _vertices;
}


template<typename Real, unsigned int Dim, unsigned int FEMSig>
PipelineMesh SolveAndExtract(
    const std::vector<PipelinePoint>& points,
    const PoissonParams& poisson_params
)
{
    typedef IsotropicUIntPack<Dim, FEMSig> Sigs;
    using solver_type = Reconstructor::Poisson::Solver<Real, Dim, Sigs>;
    using implicit_type = Reconstructor::Implicit<Real, Dim, Sigs>;

    // 1. Solution parameters
    typename Reconstructor::Poisson::SolutionParameters<Real> sParams;
    sParams.verbose = poisson_params.verbose;
    sParams.depth = poisson_params.depth;
    sParams.solveDepth = poisson_params.solveDepth;
    sParams.kernelDepth = poisson_params.kernelDepth;
    sParams.samplesPerNode = poisson_params.samplesPerNode;
    sParams.pointWeight = poisson_params.pointWeight;
    sParams.scale = poisson_params.scale;
    sParams.confidence = false;
    sParams.dirichletErode = true;
    sParams.exactInterpolation = false;
    sParams.showResidual = false;
    sParams.iters = 8;
    sParams.alignDir = 2; // Z-axis default
    sParams.baseDepth = -1;
    sParams.fullDepth = 5;
    sParams.baseVCycles = 1;

    // 2. Input stream
    using InputSampleFactory = VertexFactory::Factory<Real, VertexFactory::PositionFactory<Real, Dim>, VertexFactory::NormalFactory<Real, Dim>>;
    std::vector<typename InputSampleFactory::VertexType> inCorePoints;
    inCorePoints.reserve(points.size());
    InputSampleFactory inputSampleFactory;
    for (const auto& p : points) {
        typename InputSampleFactory::VertexType pt = inputSampleFactory();
        pt.template get<0>() = Point<Real, Dim>(p.x, p.y, p.z);
        pt.template get<1>() = Point<Real, Dim>(p.nx, p.ny, p.nz);
        inCorePoints.push_back(pt);
    }

    auto pointStream = std::make_unique<VectorBackedInputDataStream<typename InputSampleFactory::VertexType>>(inCorePoints);

    struct _InputOrientedSampleStream : public Reconstructor::InputOrientedSampleStream<Real, Dim> {
        typedef Reconstructor::Normal<Real, Dim> DataType;
        typedef DirectSum<Real, Reconstructor::Position<Real, Dim>, DataType> SampleType;
        typedef InputDataStream<SampleType> _InputPointStream;
        _InputPointStream &pointStream;
        SampleType scratch;
        _InputOrientedSampleStream(_InputPointStream &pointStream)
            : pointStream(pointStream), scratch(Reconstructor::Position<Real, Dim>(), Reconstructor::Normal<Real, Dim>()) {}
        void reset(void) override { pointStream.reset(); }
        bool read(Reconstructor::Position<Real, Dim> &p, Reconstructor::Normal<Real, Dim> &n) override {
            bool ret = pointStream.read(scratch);
            if (ret) p = scratch.template get<0>(), n = scratch.template get<1>();
            return ret;
        }
        bool read(unsigned int thread, Reconstructor::Position<Real, Dim> &p, Reconstructor::Normal<Real, Dim> &n) override {
            bool ret = pointStream.read(thread, scratch);
            if (ret) p = scratch.template get<0>(), n = scratch.template get<1>();
            return ret;
        }
    };

    _InputOrientedSampleStream sampleStream(*pointStream);
    typename Reconstructor::Poisson::EnvelopeMesh<Real, Dim>* envelopeMesh = nullptr;

    // 3. Solve
    implicit_type* implicit = solver_type::Solve(sampleStream, sParams, envelopeMesh);

    // 4. Extract Level Set
    Reconstructor::LevelSetExtractionParameters meParams;
    meParams.linearFit = false;
    meParams.polygonMesh = false;
    meParams.outputDensity = true;
    meParams.verbose = poisson_params.verbose;

    using OutputFactory = VertexFactory::Factory<Real, VertexFactory::PositionFactory<Real, Dim>, VertexFactory::ValueFactory<Real>>;
    OutputFactory factory = Reconstructor::OutputVertexInfo<Real, Dim, false, true>::GetFactory();
    Reconstructor::OutputInputFactoryTypeStream<Real, Dim, OutputFactory, true, true> vertexStream(
        factory, Reconstructor::OutputVertexInfo<Real, Dim, false, true>::Convert
    );
    Reconstructor::OutputInputFaceStream<Dim - 1, true, true> faceStream;

    implicit->extractLevelSet(vertexStream, faceStream, meParams);

    // 5. Retrieve mesh into vectors
    vertexStream.reset();
    faceStream.reset();

    std::vector<typename OutputFactory::VertexType> vertices(vertexStream.size());
    for (size_t i = 0; i < vertices.size(); ++i) {
        vertices[i] = factory();
        if (!vertexStream.read(vertices[i])) {
            throw std::runtime_error("Failed to read vertex from stream");
        }
    }

    std::vector<std::vector<int>> polygons(faceStream.size());
    for (size_t i = 0; i < polygons.size(); ++i) {
        std::vector<node_index_type> f;
        if (!faceStream.read(f)) {
            throw std::runtime_error("Failed to read face from stream");
        }
        polygons[i].resize(f.size());
        for (size_t j = 0; j < f.size(); ++j) {
            polygons[i][j] = (int)f[j];
        }
    }

    delete implicit;

    // Format output PipelineMesh directly (no trimming in poisson stage)
    PipelineMesh out_mesh;
    out_mesh.vertices.reserve(vertices.size());
    for (const auto& v : vertices) {
        auto pos = v.template get<0>();
        out_mesh.vertices.push_back({pos[0], pos[1], pos[2], v.template get<1>()});
    }
    out_mesh.faces = polygons;

    return out_mesh;
}

} // namespace

PipelineMesh run_poisson_reconstruction(
    const std::vector<PipelinePoint>& points,
    const PoissonParams& poisson_params
)
{
    BoundaryType b_type;
    switch(poisson_params.bType) {
        case 1: b_type = BOUNDARY_FREE; break;
        case 2: b_type = BOUNDARY_NEUMANN; break;
        case 3: b_type = BOUNDARY_DIRICHLET; break;
        default: throw std::runtime_error("Invalid boundary type");
    }

    if (poisson_params.degree == 1) {
        if (b_type == BOUNDARY_FREE) {
            return SolveAndExtract<double, 3, FEMDegreeAndBType<1, BOUNDARY_FREE>::Signature>(points, poisson_params);
        } else if (b_type == BOUNDARY_NEUMANN) {
            return SolveAndExtract<double, 3, FEMDegreeAndBType<1, BOUNDARY_NEUMANN>::Signature>(points, poisson_params);
        } else if (b_type == BOUNDARY_DIRICHLET) {
            return SolveAndExtract<double, 3, FEMDegreeAndBType<1, BOUNDARY_DIRICHLET>::Signature>(points, poisson_params);
        }
    } else if (poisson_params.degree == 2) {
        if (b_type == BOUNDARY_FREE) {
            return SolveAndExtract<double, 3, FEMDegreeAndBType<2, BOUNDARY_FREE>::Signature>(points, poisson_params);
        } else if (b_type == BOUNDARY_NEUMANN) {
            return SolveAndExtract<double, 3, FEMDegreeAndBType<2, BOUNDARY_NEUMANN>::Signature>(points, poisson_params);
        } else if (b_type == BOUNDARY_DIRICHLET) {
            return SolveAndExtract<double, 3, FEMDegreeAndBType<2, BOUNDARY_DIRICHLET>::Signature>(points, poisson_params);
        }
    }
    throw std::runtime_error("Only B-Splines of degree 1 - 2 are supported");
}

PipelineMesh trim_mesh(
    const PipelineMesh& input_mesh,
    const TrimParams& trim_params
)
{
    static constexpr unsigned int Dim = 3;
    using Real = double;

    using OutputFactory = VertexFactory::Factory<Real, VertexFactory::PositionFactory<Real, Dim>, VertexFactory::ValueFactory<Real>>;
    OutputFactory factory;

    // Convert input_mesh to the format expected by SplitPolygon
    std::vector<typename OutputFactory::VertexType> vertices(input_mesh.vertices.size());
    for (size_t i = 0; i < vertices.size(); ++i) {
        vertices[i] = factory();
        vertices[i].template get<0>() = Point<Real, Dim>(input_mesh.vertices[i].x, input_mesh.vertices[i].y, input_mesh.vertices[i].z);
        vertices[i].template get<1>() = input_mesh.vertices[i].value;
    }

    std::vector<std::vector<int>> polygons = input_mesh.faces;

    std::unordered_map<EdgeKey<int>, int, typename EdgeKey<int>::Hasher> vertexTable;
    std::vector<std::vector<int>> ltPolygons, gtPolygons;
    std::vector<bool> ltFlags, gtFlags;

    for (size_t i = 0; i < polygons.size(); ++i) {
        SplitPolygon<Real, Dim, int, typename OutputFactory::VertexType>(
            polygons[i], vertices, &ltPolygons, &gtPolygons, &ltFlags, &gtFlags, vertexTable, (Real)trim_params.trimThreshold
        );
    }

    // Island removal
    if (trim_params.islandAreaRatio > 0) {
        std::vector<std::vector<int>> _polygons, _components;
        size_t gtComponentStart;
        {
            std::vector<std::vector<int>> ltComponents, gtComponents;
            SetConnectedComponents(ltPolygons, ltComponents);
            SetConnectedComponents(gtPolygons, gtComponents);
            gtComponentStart = ltComponents.size();
            for (unsigned int i = 0; i < gtComponents.size(); i++) {
                for (unsigned int j = 0; j < gtComponents[i].size(); j++) {
                    gtComponents[i][j] += (int)ltPolygons.size();
                }
            }

            _polygons.reserve(ltPolygons.size() + gtPolygons.size());
            _components.reserve(ltComponents.size() + gtComponents.size());
            _polygons.insert(_polygons.end(), ltPolygons.begin(), ltPolygons.end());
            _polygons.insert(_polygons.end(), gtPolygons.begin(), gtPolygons.end());
            _components.insert(_components.end(), ltComponents.begin(), ltComponents.end());
            _components.insert(_components.end(), gtComponents.begin(), gtComponents.end());
        }
        std::vector<typename ComponentGraph<int>::Node> nodes(_components.size());

        for (unsigned int i = 0; i < _components.size(); i++) {
            nodes[i].polygonIndices.insert(nodes[i].polygonIndices.end(), _components[i].begin(), _components[i].end());
            for (auto iter = nodes[i].polygonIndices.begin(); iter != nodes[i].polygonIndices.end(); iter++) {
                nodes[i].area += PolygonArea<Real, Dim, int, typename OutputFactory::VertexType>(vertices, _polygons[*iter]);
            }
        }

        std::unordered_map<HalfEdgeKey<int>, int, typename HalfEdgeKey<int>::Hasher> componentBoundaryHalfEdges;
        for (unsigned int i = 0; i < _components.size(); i++) {
            std::unordered_set<HalfEdgeKey<int>, typename HalfEdgeKey<int>::Hasher> componentHalfEdges;
            for (unsigned int j = 0; j < _components[i].size(); j++) {
                const std::vector<int>& poly = _polygons[_components[i][j]];
                for (unsigned int k = 0; k < poly.size(); k++) {
                    int v1 = poly[k], v2 = poly[(k + 1) % poly.size()];
                    HalfEdgeKey<int> eKey = HalfEdgeKey<int>(v1, v2);
                    componentHalfEdges.insert(eKey);
                }
            }
            for (auto iter = componentHalfEdges.begin(); iter != componentHalfEdges.end(); iter++) {
                HalfEdgeKey<int> key = *iter;
                HalfEdgeKey<int> _key = key.opposite();
                if (componentHalfEdges.find(_key) == componentHalfEdges.end()) {
                    componentBoundaryHalfEdges[key] = (int)i;
                }
            }
        }

        std::unordered_set<EdgeKey<int>, typename EdgeKey<int>::Hasher> componentEdges;
        for (auto iter = componentBoundaryHalfEdges.begin(); iter != componentBoundaryHalfEdges.end(); iter++) {
            HalfEdgeKey<int> key = iter->first;
            HalfEdgeKey<int> _key = key.opposite();
            auto _iter = componentBoundaryHalfEdges.find(_key);
            if (_iter != componentBoundaryHalfEdges.end()) {
                componentEdges.insert(EdgeKey<int>(iter->second, _iter->second));
            }
        }
        for (auto iter = componentEdges.begin(); iter != componentEdges.end(); iter++) {
            nodes[iter->key1].neighbors.push_back(&nodes[iter->key2]);
            nodes[iter->key2].neighbors.push_back(&nodes[iter->key1]);
        }

        double area = 0;
        for (unsigned int i = 0; i < nodes.size(); i++) area += nodes[i].area;

        bool done = false;
        while (!done) {
            done = true;
            unsigned int idx = -1;
            for (unsigned int i = 0; i < nodes.size(); i++) {
                if (nodes[i].polygonIndices.size() && nodes[i].neighbors.size()) {
                    if (idx == -1 || nodes[i].area < nodes[idx].area) idx = i;
                }
            }
            if (idx != -1 && nodes[idx].area < area * trim_params.islandAreaRatio) {
                nodes[idx].merge();
                done = false;
            }
        }

        ltPolygons.clear();
        gtPolygons.clear();

        for (unsigned int i = 0; i < gtComponentStart; i++) {
            if (!nodes[i].neighbors.size() && nodes[i].area < area * trim_params.islandAreaRatio && trim_params.removeIslands) ;
            else {
                for (auto iter = nodes[i].polygonIndices.begin(); iter != nodes[i].polygonIndices.end(); iter++) {
                    ltPolygons.push_back(_polygons[*iter]);
                }
            }
        }
        for (unsigned int i = (unsigned int)gtComponentStart; i < nodes.size(); i++) {
            if (!nodes[i].neighbors.size() && nodes[i].area < area * trim_params.islandAreaRatio && trim_params.removeIslands) ;
            else {
                for (auto iter = nodes[i].polygonIndices.begin(); iter != nodes[i].polygonIndices.end(); iter++) {
                    gtPolygons.push_back(_polygons[*iter]);
                }
            }
        }
    }

    // Triangulate gtPolygons
    std::vector<std::vector<int>> triangles;
    Triangulate<Real, Dim, int, typename OutputFactory::VertexType>(vertices, gtPolygons, triangles);

    RemoveHangingVertices<int, typename OutputFactory::VertexType>(vertices, triangles);

    // Format output PipelineMesh
    PipelineMesh out_mesh;
    out_mesh.vertices.reserve(vertices.size());
    for (const auto& v : vertices) {
        auto pos = v.template get<0>();
        out_mesh.vertices.push_back({pos[0], pos[1], pos[2], v.template get<1>()});
    }
    out_mesh.faces = triangles;

    return out_mesh;
}
