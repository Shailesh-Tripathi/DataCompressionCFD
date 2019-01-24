//#include<bits/stdc++.h>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <tuple>
#include <cstring>
#include <math.h>
#include <algorithm>

#include "mkl.h"

using namespace std;

/* To avoid constantly repeating the part of code that checks inbound SparseBLAS functions' status,
 *    use macro CALL_AND_CHECK_STATUS */
#define CALL_AND_CHECK_STATUS(function) do { 														\
          if(function != SPARSE_STATUS_SUCCESS)										             	\
          {                                    										             	\
			std::string message;																	\
			switch(function)																		\
			{																						\
				case SPARSE_STATUS_NOT_INITIALIZED  : message = " SPARSE_STATUS_NOT_INITIALIZED\n";	\
													  break;										\
				case SPARSE_STATUS_ALLOC_FAILED     : message = " SPARSE_STATUS_ALLOC_FAILED\n";    \
													  break;										\
				case SPARSE_STATUS_INVALID_VALUE    : message = " SPARSE_STATUS_INVALID_VALUE\n";   \
													  break;										\
				case SPARSE_STATUS_EXECUTION_FAILED : message = " SPARSE_STATUS_EXECUTION_FAILED\n";\
													  break;										\
				case SPARSE_STATUS_INTERNAL_ERROR   : message = " SPARSE_STATUS_INTERNAL_ERROR\n";  \
													  break;										\
				case SPARSE_STATUS_NOT_SUPPORTED    : message = " SPARSE_STATUS_NOT_SUPPORTED\n";   \
													  break;										\
				default : message = "UNKNOWN_ERROR\n";												\
			}																						\
          	printf("Error message=%s", message.c_str()); fflush(0);     									\
          	exit(0);		                                    									\
          }                                                											\
} while(0)


// enum for triangular matrix type
enum TriangleType {LOWER, UPPER};

//enum for Sparse Matrix type
enum SparseMatrixType {CSR, CSC};

/******************************************************************************/
// Generic class to store sparse matrix irrespective of CSR/CSC representation.
template <class T>
class SparseMatrix
{
    public:
		// sparseMat handle
        sparse_matrix_t sparseMat;

		// Descriptor for sparse matrix
        struct matrix_descr descrMat;

		// Type of sparse matrix : CSR or CSC
		SparseMatrixType CSType;

        MKL_INT numRows, numCols, nnz;
        vector<T> val; //values
        vector<MKL_INT> ptr; // rowPtr for CSR
						 // colPtr for CSC

        vector<MKL_INT> ind; // column indices for CSR
						 // row indices for CSC

		// Constructor
        SparseMatrix (int r = 0, int c = 0, SparseMatrixType type = CSR):numRows(r) , numCols(c), CSType(type)
        {
            val.clear();

			if(CSR == CSType)
            	ptr.resize(numRows+1);
			else
				ptr.resize(numCols+1);

            ptr[0]=0;
            ind.clear();
			descrMat.type = SPARSE_MATRIX_TYPE_GENERAL;
			
			// Create the handle by default
//			createHandle();
        }

		// Utility function to add a row in the last Row in a CSR matrix
        void addDenseInLastRow(vector<T> row)
        {
			// This operation is only allowed in matrix represented by CSR representation.
			if(CSR != CSType)
			{
				std::cout << "Add row is allowed only in CSR matrix\n";
				exit(0);
			}

            numRows++;
            for(int i = 0; i < row.size(); i++)
            {
                if((T)0 != row[i])
                {	
					val.push_back(row[i]);
					ind.push_back(i);
				}
            }
            
			ptr.push_back(val.size());
            // Update NNZ
            nnz = val.size();
        }

        // Increment number of columns
        void incrementColumnSize(int n)
        {
            numCols += n;
        }
    
        // Create handle with matrix stored in CSR format
        // This will be called everytime the matrix is updated.
        void createHandle()
		{
			// First create argument list and then pass it in appropriate function.
			auto args = make_tuple( (sparse_matrix_t*) &sparseMat,
                                    (sparse_index_base_t) SPARSE_INDEX_BASE_ZERO,
                                    (MKL_INT) numRows,
                                    (MKL_INT) numCols,
                                    (MKL_INT *) &ptr[0],
                                    (MKL_INT *) &ptr[0]+1,
                                    (MKL_INT *) &ind[0],
                                    (T *) &val[0] );

			// check the matrix type
			if(CSR == CSType)
			{
            	// Check if single or double precision
            	if(std::is_same<T, float>::value)
            	{
            	    CALL_AND_CHECK_STATUS(std::apply(mkl_sparse_s_create_csr, args));
            	}
//            	else
//            	{
//            	    std::apply(mkl_sparse_d_create_csr, args);
//            	}
			}
			else // CSC representation
			{
            	// Check if single or double precision
            	if(std::is_same<T, float>::value)
            	{
            	    CALL_AND_CHECK_STATUS(std::apply(mkl_sparse_s_create_csc, args));
            	}
//            	else
//            	{
//            	    std::apply(mkl_sparse_d_create_csc, args);
//            	}
			}
			
        }

		void setNumRows(int r)
		{
			numRows = r;
		}

		void setNumCols(int c)
		{
			numCols = c;
		}

		void setNnz(int n)
		{
			nnz = n;
		}

		void setCSType(SparseMatrixType t)
		{
			CSType = t;
		}

		// Utility function to print sparse matrix vectors
		void print()
		{
			std::cout << "Matrix type: 0(CSR) or 1(CSC) " << CSType;
			std::cout << "\nValues:\n";
			for(auto v:val)
			{
				std::cout << v << ' ';
			}
			
			std::cout << "\nIndices:\n";
			for(auto i:ind)
			{
				std::cout << i << ' ';
			}
			
			std::cout << "\nPtr:\n";
			for(auto p:ptr)
			{
				std::cout << p << ' ';
			}
		}
		
};

// Utility function to print vector
template <typename T>
void printVector(vector<T> vec)
{
	std::cout << "Printing vector:\n";
	for(auto v:vec)
		std::cout << v << ' ';
	std::cout << std::endl;
}

/******************************************************************************/
// Utility function for dot product of two dense vectors
template <typename T>
T dotProduct(vector<T> &vecA,
			 vector<T> &vecB)
{
	if(vecA.size() != vecB.size())
	{
		std::cout << "Size of two vectors not equal for dot product.";
		exit(0);
	}

	T result(0);
	
	// Create argument list first
	auto args = std::make_tuple(vecA.size(),
                                &vecA[0],
                                1,
                                &vecB[0],
                                1);

	// Compute Dot Product
    if(std::is_same<T, float>::value)
	{	
		result = std::apply(cblas_sdot, args);
	}
//	else
//	{	
//		result = std::apply(cblas_ddot, args);
//	}

	return result;
}

/******************************************************************************/
template <typename T>
void sparseMatVecMul(SparseMatrix<T>& mat,
					 vector<T>& input,
					 vector<T>& output,
					 sparse_operation_t operation)
{
	// Make argument list
	auto args = make_tuple(operation,
                           1.0,
                           mat.sparseMat,
                           mat.descrMat,
                           &input[0],
                           0.0,
                           &output[0]);

    if(std::is_same<T, float>::value)
	{
		CALL_AND_CHECK_STATUS(std::apply(mkl_sparse_s_mv, args));
	}
//	else
//	{
//		std::apply(mkl_sparse_d_mv, args);
//	}
}

/******************************************************************************/
template <typename T>
void solveTriangular(SparseMatrix<T>& mat,
					 vector<T>& input,
					 vector<T>& output,
					 sparse_operation_t operation,
					 TriangleType type)
{
	struct matrix_descr descr;

	// Temporary descriptor for triangular matrix
	descr.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
	descr.diag = SPARSE_DIAG_NON_UNIT;
	descr.mode = (type == LOWER) ? SPARSE_FILL_MODE_LOWER : SPARSE_FILL_MODE_UPPER;

	// Argument list
	auto args = make_tuple( operation,
                            1.0,
                            mat.sparseMat,
                            descr,
                            &input[0],
                            &output[0] );

    if(std::is_same<T, float>::value)
	{
		CALL_AND_CHECK_STATUS(std::apply( mkl_sparse_s_trsv, args));
	}
//	else
//	{
//		std::apply( mkl_sparse_d_trsv, args);
//	}
}

/******************************************************************************/
// This function will compute the selected_g_vector for current iteration which does not
// includes the position selected in same iteration. But G_partial will be using that
// selected index. 
template <typename T>
void selectColumnAndUpdateG_partial(const int columnIndex,
									const vector<int>& selectedRowIndices,
									const SparseMatrix<T>& G,
									SparseMatrix<T>& G_partial,
									vector<T>& selected_g_vector)
{
	if(G.CSType != CSC)
	{
		std::cout << "G needs to be a CSC matrix\n";
		exit(0);
	}

	int startIndx = G.ptr[columnIndex];
	int endIndx = G.ptr[columnIndex+1];

	// Append this column on the right of G_partial
	G_partial.val.insert(G_partial.val.end(), G.val.begin()+startIndx, G.val.begin()+endIndx);
	G_partial.ind.insert(G_partial.ind.end(), G.ind.begin()+startIndx, G.ind.begin()+endIndx);
	G_partial.ptr.push_back(G_partial.val.size());
	G_partial.numCols++;
	G_partial.createHandle();

	//////////////////////////////////////////////////////////////////////////////
	// Create dense selected_g_vector from sparse.
	// Reset all the vector values to zero.
	vector<T> tempGColumn(G.numCols, 0.0);

	// Select only the required indices	
	for(int i = startIndx; i < endIndx; i++)
	{
		tempGColumn[G.ind[i]] = G.val[i];
	}

	// TODO: This might be optimized to use less memory. Use a hashmap to instead of full vector.
	// Increase size of selected_g_vector by 1. Hence add a dummy value at the end.
	selected_g_vector.push_back(0.0);
	// NOTE: We do not iterate over the last element of selectedRowIndices because that is not
	// 		 required for current iteration.
	// Set the values in the selected_g_vector.
	for(int i = 0; i < selectedRowIndices.size() - 1; i++)
	{
		selected_g_vector[i] = tempGColumn[selectedRowIndices[i]];
	}
}

/******************************************************************************/
//Utility function for Addition
template <typename T>
void vectorAdd( const T coeff1,
				const vector<T>& input1,
				const T coeff2,
				const vector<T>& input2,
				vector<T>& output )
{
	if(input1.size() != input2.size() || input1.size() != output.size())
	{
		std::cout << " Size of input1, input2 and output should be eqaul.\n";
		exit(0);
	}

	// Reset output to 0
	std::fill(output.begin(), output.end(), 0.0);

	auto args1 = make_tuple(input1.size(), coeff1, &input1[0], 1, &output[0], 1);
	auto args2 = make_tuple(input2.size(), coeff2, &input2[0], 1, &output[0], 1);

    if(std::is_same<T, float>::value)
	{
		std::apply(cblas_saxpy, args1);
		std::apply(cblas_saxpy, args2);
	}
//	else
//	{
//		std::apply(cblas_daxpy, args);
//	}
}


/******************************************************************************/
template <typename T>
void setSelectedIndices( vector<T>& CoefficientDense,
						 vector<int>& selectedIndices,
						 vector<T>& coeffSparse )
{
	for(int i = 0; i < selectedIndices.size(); i++)
	{
		CoefficientDense[selectedIndices[i]] = coeffSparse[i];
	}
}

/******************************************************************************/
template <typename T>
vector<T> getSelectedIndices( vector<T>& input,
							  vector<int>& indices )
{
	vector<T> result;
	for(auto it:indices)
		result.push_back(input[it]);

	return result;
}

/******************************************************************************/
// Utility function to get max absolute element position
template <typename T>
int findMaxAbsolute(const vector<T>& alpha)
{
	// Declare a lambda function to compare absolute values
	auto absoluteComparator = [](T a, T b) {return abs(a) < abs(b);};

	return max_element(alpha.begin(), alpha.end(), absoluteComparator) - alpha.begin();
}

/******************************************************************************/
template <typename T>
void batchOMPCholesky(SparseMatrix<T>& dictionary,
					  SparseMatrix<T>& G,
					  SparseMatrix<T>& G_partial,
					  vector<T>& data,
					  vector<T>& selected_g_vector,
					  vector<T>& coefficientFinal, 
					  int numPoints,
					  int numBasis,
					  float target_error,
					  int max_iterations)
{
    vector<T> alpha(numBasis, 0.0); // Projection of data on remaining dictionary
	vector<T> alpha0(numBasis, 0.0);
	vector<T> beta(numBasis, 0.0);

    // alpha0 = Dict' * Data
    // sparse matrix vector multiplication
    sparseMatVecMul(dictionary, data, alpha0, SPARSE_OPERATION_TRANSPOSE);
    
    alpha = alpha0;
    //memcpy(alpha, alpha0, numBasis * sizeof(*alpha0));
    
    // norm_x = sqrt(Data' * Data)
    T normData = sqrt(dotProduct(data, data));

    float error(1.0);

    // The number of active basis vectors will increment at each iteration. Hence, start a prediction of 1 and keep doubling as the size reaches the predicted value.
    int currentNumberOfActiveBasis, predictedNumberOfActiveBasis;
    
//    vector<T> coefficientFinal(numBasis, 0.0);

    vector<int> selectedIndices; //I
    
    SparseMatrix<T>  lowerTriangularMat(0,0,CSR);

    lowerTriangularMat.incrementColumnSize(1);
    lowerTriangularMat.addDenseInLastRow(vector<T> {1.0});
	lowerTriangularMat.createHandle();

    T delta_previous(0.0);
    int iterationCount(1);
    T error_sq(normData*normData);

    T tolerance(1e-6); // Tolerance value to stop selection when linearly independent modes are exhausted.

    // coefficients vector to be updated at each iteration    
    vector<T> tempCoeff;
    vector<T> tempSolution;

	vector<T> wVector(1, 0.0); //The size of this will increase by 1 at each iteration.

    // Continue iteration until target error is not reached
    while(error > target_error)
    {
        // Find the position of max projection
        int position = findMaxAbsolute(alpha);
        
        T maxVal = abs(alpha[position]);

        // Break the selection process if the selected basis is linearly dependent on already selected basis
        if(maxVal < tolerance)
            break;

		// Append the last selected index.
		selectedIndices.push_back(position);

		// Select the 'position' column and append it to G_partial's last column
		selectColumnAndUpdateG_partial(position,
									   selectedIndices,
									   G,
									   G_partial,
									   selected_g_vector);

		// Append dummy value to increase size by 1.
		//wVector.push_back(0.0);
		tempCoeff.push_back(0.0);
		tempSolution.push_back(0.0);

        if(iterationCount > 1)
        {
            solveTriangular( lowerTriangularMat,
							 selected_g_vector,
							 wVector,
							 SPARSE_OPERATION_NON_TRANSPOSE,
							 LOWER);
			
			// Compute norm
            T wNormSquare = dotProduct(wVector, wVector);
            wVector.push_back(sqrt(1.0 - wNormSquare));

            lowerTriangularMat.incrementColumnSize(1);
            lowerTriangularMat.addDenseInLastRow(wVector);
			lowerTriangularMat.createHandle();
        }
		else
			selected_g_vector.clear();

		// Store selected alpha values in temporary vector	
		auto tempSelectedAlpha = getSelectedIndices(alpha0, selectedIndices);
		
        // Solve for coefficients in a two step process.
        solveTriangular( lowerTriangularMat,
						 tempSelectedAlpha,
						 tempSolution,
						 SPARSE_OPERATION_NON_TRANSPOSE,
						 LOWER );

		tempSelectedAlpha.clear();

        solveTriangular( lowerTriangularMat,
						 tempSolution,
						 tempCoeff,
						 SPARSE_OPERATION_TRANSPOSE,
						 LOWER );

        // Compute beta
        // Sparse matrix-vector multiplication
        sparseMatVecMul(G_partial,
						tempCoeff,
						beta,
						SPARSE_OPERATION_NON_TRANSPOSE );
        
        // Vector substraction
		// aplha = alpha0 - beta
        vectorAdd( 1.0f,    // Coeff-1
				   alpha0, // Input-1
				   -1.0f,   // Coeff-2
				   beta,   // Input-2
				   alpha); // Output

		// Store selected beta in temporary array
		auto tempSelectedBeta = getSelectedIndices(beta, selectedIndices);
	
        // Compute Error for current iteration using error from previous iteration
        T delta_current = dotProduct(tempCoeff, tempSelectedBeta);
		tempSelectedBeta.clear();

        error_sq = error_sq - delta_current + delta_previous;

        error = sqrt(error_sq) / normData;

        // update previous delta
        delta_previous = delta_current;

        // Increment iteration count
        iterationCount++;
            
		std::cout << "iter-" << iterationCount << " Error=" << error << std::endl;
    }    

    setSelectedIndices(coefficientFinal, selectedIndices, tempCoeff);
}

/******************************************************************************/
// Utility functions for each data type to extract values in a line
int parseInt(ifstream& fin)
{
	string label;
	int val;

	fin >> label >> val;
	
	std::cout << val << std::endl;
	return val;
}

/******************************************************************************/
float parseFloat(ifstream& fin)
{
	string label;
	float val;

	fin >> label >> val;
	
	std::cout << val << std::endl;
	return val;
}

/******************************************************************************/
std::string parseString(ifstream& fin)
{
	std::string label;
	std::string val;

	fin >> label >> val;
	
	std::cout << val << std::endl;
	return val;
}

/******************************************************************************/
// Parse input file to get all the parameters for the code
void readInputFile(const string inputFileName,
				   string& GMatrixFileName,
				   string& dataFileName,
				   string& dictionaryFileName,
			 	   int& numPoints,
				   int& numBasis,
				   int& maxIterations,
				   float& targetError,
				   int& G_nnz,
				   int& dictionary_nnz)
{
	ifstream fin(inputFileName);
	
	// Check if file opened
	if(!fin.is_open())
	{
		std::cout << "Error opening file!\n";
		exit(0);
	}

	// Parse these in order
	GMatrixFileName = parseString(fin);
	dataFileName = parseString(fin);
	dictionaryFileName = parseString(fin);
	numPoints = parseInt(fin);
	numBasis = parseInt(fin);
	maxIterations = parseInt(fin);
	targetError = parseFloat(fin);
	G_nnz = parseInt(fin);
	dictionary_nnz = parseInt(fin);

	fin.close();
}

/******************************************************************************/
void readSparseMatrixFromFile( const std::string& fileName,
							   const int numRows,
							   const int numCols,
							   const int nnz,
							   const SparseMatrixType type,
							   SparseMatrix<float>& spMat )
{
	ifstream fin(fileName);

	if(!fin.is_open())
	{
		std::cout << "Error opening file!\n";
		exit(0);
	}

	spMat.setNumRows(numRows);
	spMat.setNumCols(numCols);
	spMat.setNnz(nnz);
	spMat.setCSType(type);

	spMat.val.resize(nnz);
	spMat.ind.resize(nnz);

	// Size of ptr depends on CSType
	if(CSR == type)
		spMat.ptr.resize(numRows + 1);
	else
		spMat.ptr.resize(numCols + 1);

	// Read all the non-zero values
	for(int i = 0; i < nnz; i++)
	{
		fin >> spMat.val[i];
	}

	// Read Indx
	for(int i = 0; i < nnz; i++)
	{
		fin >> spMat.ind[i];
	}
		
	// Read start pointers
	for(int i = 0; i < spMat.ptr.size(); i++)
	{
		fin >> spMat.ptr[i];
	}

	spMat.createHandle();
	fin.close();
}

/******************************************************************************/
void readDataVectorFromFile( std::string dataFileName,
							 int numPoints,
							 vector<float>& data )
{
	ifstream fin(dataFileName);
	
	if(!fin.is_open())
	{
		std::cout << "Error opening file!\n";
		exit(0);
	}

	// Resize data vector
	data.resize(numPoints);

	for(int i = 0; i < numPoints; i++)
	{
		fin >> data[i];
	}
	
	fin.close();
}


/******************************************************************************/
int main()
{
	int numPoints, numBasis;
	int maxIterations;
	float targetError;
	int G_nnz, dictionary_nnz; 

	std::string inputFileName("input.txt"), GMatrixFileName, dataFileName, dictionaryFileName;	

	readInputFile( inputFileName,
				   GMatrixFileName,
				   dataFileName,
				   dictionaryFileName,
			 	   numPoints,
				   numBasis,
				   maxIterations,
				   targetError,
				   G_nnz,
				   dictionary_nnz );

    SparseMatrix<float> dictionarySparse; // Dictionary
    std::vector<float> data; //Data to be reconstructed
    SparseMatrix<float> GSparse(numBasis, numBasis, CSC); // G = Dict' * Dict ; This will be reconstructed globally
   	SparseMatrix<float> GPartialSparse(numBasis, 0, CSC);
       
	// g vector which is a subvector from G matrix
	vector<float> selected_g_vector;
    vector<float> coefficientFinal(numBasis, 0.0);

	// Compute G = Dict' * Dict 
    // computeGmatrix(dictionary, G, num_points, numBasis);

	// NOTE: Read G matrix from file for now.
	readSparseMatrixFromFile( GMatrixFileName,
							  numBasis,
							  numBasis,
							  G_nnz,
							  CSC,
							  GSparse );

	// Read Dictionary from file	
	readSparseMatrixFromFile( dictionaryFileName,
							  numPoints,
							  numBasis,
							  dictionary_nnz,
							  CSC,
							  dictionarySparse );

	// Read single data vector from file
	readDataVectorFromFile( dataFileName,
						    numPoints,
						    data );	
	

	// call batch OMP cholesky
	batchOMPCholesky( dictionarySparse,
					  GSparse,
					  GPartialSparse,
					  data,
					  selected_g_vector,
					  coefficientFinal,
					  numPoints,
					  numBasis,
					  targetError,
					  maxIterations );

	// Print the result
	for(int i = 0; i < coefficientFinal.size(); i++)
	{
		std::cout << i << ' ' << coefficientFinal[i] << std::endl;
	}			  
}
