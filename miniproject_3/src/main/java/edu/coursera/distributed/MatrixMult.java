package edu.coursera.distributed;

import edu.coursera.distributed.util.MPI;
import edu.coursera.distributed.util.MPI.MPIException;

/**
 * A wrapper class for a parallel, MPI-based matrix multiply implementation.
 */
public class MatrixMult {
    /**
     * A parallel implementation of matrix multiply using MPI to express SPMD
     * parallelism. In particular, this method should store the output of
     * multiplying the matrices a and b into the matrix c.
     *
     * This method is called simultaneously by all MPI ranks in a running MPI
     * program. For simplicity MPI_Init has already been called, and
     * MPI_Finalize should not be called in parallelMatrixMultiply.
     *
     * On entry to parallelMatrixMultiply, the following will be true of a, b,
     * and c:
     *
     *   1) The matrix a will only be filled with the input values on MPI rank
     *      zero. Matrix a on all other ranks will be empty (initialized to all
     *      zeros).
     *   2) Likewise, the matrix b will only be filled with input values on MPI
     *      rank zero. Matrix b on all other ranks will be empty (initialized to
     *      all zeros).
     *   3) Matrix c will be initialized to all zeros on all ranks.
     *
     * Upon returning from parallelMatrixMultiply, the following must be true:
     *
     *   1) On rank zero, matrix c must be filled with the final output of the
     *      full matrix multiplication. The contents of matrix c on all other
     *      ranks are ignored.
     *
     * Therefore, it is the responsibility of this method to distribute the
     * input data in a and b across all MPI ranks for maximal parallelism,
     * perform the matrix multiply in parallel, and finally collect the output
     * data in c from all ranks back to the zeroth rank. You may use any of the
     * MPI APIs provided in the mpi object to accomplish this.
     *
     * A reference sequential implementation is provided below, demonstrating
     * the use of the Matrix class's APIs.
     *
     * @param a Input matrix
     * @param b Input matrix
     * @param c Output matrix
     * @param mpi MPI object supporting MPI APIs
     * @throws MPIException On MPI error. It is not expected that your
     *                      implementation should throw any MPI errors during
     *                      normal operation.
     */
    public static void parallelMatrixMultiply(Matrix a, Matrix b, Matrix c,
                                              final MPI mpi) throws MPIException {
        int numRanks = mpi.MPI_Comm_size(mpi.MPI_COMM_WORLD);
        int myRank = mpi.MPI_Comm_rank(mpi.MPI_COMM_WORLD);
        if (myRank==0) {
            Matrix[] allResults = new Matrix[numRanks];
            int[] rowSizes = new int[numRanks];
            for (int i = 0; i<numRanks; i++) {
                int rowStart = getRowStart(i,numRanks,a.getNRows());
                int rowSize = getRowCount(i,numRanks,a.getNRows());
                rowSizes[i] = rowSize;
                allResults[i] = new Matrix(rowSize,b.getNCols());
                if (i!=0) {
                    mpi.MPI_Send(a.getValues(), a.getOffsetOfRow(rowStart), a.getNCols()*rowSize, i, 0, mpi.MPI_COMM_WORLD);
                    mpi.MPI_Send(b.getValues(), 0, b.getNRows()*b.getNCols(), i, 0, mpi.MPI_COMM_WORLD);
                }
                else {
                    Matrix zeroRows = new Matrix(rowSize,a.getNCols());
                    for (int k = 0; k<rowSize; k++) {
                        for (int l = 0; l<a.getNCols(); l++) {
                            zeroRows.set(k,l,a.getValues()[a.getOffsetOfRow(k)+l]);
                        }
                    }
                    seqMatrixMultiply(zeroRows,b,allResults[0]);
                }
            }
            for (int i = 1; i<allResults.length; i++) {
                getInputIntoMatrix(allResults[i],i,mpi);
            }
            int curRow = 0;
            for (int i = 0; i<numRanks; i++) {
                for (int j = 0; j<allResults[i].getNRows(); j++) {
                    for (int k = 0; k < b.getNCols(); k++) {
                        c.set(curRow, k, allResults[i].get(j,k));
                    }
                    curRow+=1;
                }
            }
        }
        else {
            int numRows = getRowCount(myRank,numRanks,a.getNRows());
            Matrix input1 = new Matrix(numRows,a.getNCols());
            Matrix input2 = new Matrix(b.getNRows(),b.getNCols());
            Matrix result = new Matrix(numRows,b.getNCols());
            getInputIntoMatrix(input1,0,mpi);
            getInputIntoMatrix(input2,0,mpi);
            seqMatrixMultiply(input1,input2,result);
            mpi.MPI_Send(result.getValues(),0,result.getNRows()*result.getNCols(),0,0,mpi.MPI_COMM_WORLD);
        }
    }

    private static void seqMatrixMultiply(Matrix a, Matrix b, Matrix c) {
        for (int i = 0; i < c.getNRows(); i++) {
            for (int j = 0; j < c.getNCols(); j++) {
                c.set(i, j, 0.0);
                for (int k = 0; k < b.getNRows(); k++) {
                    c.incr(i, j, a.get(i, k) * b.get(k, j));
                }
            }
        }
    }

    private static void getInputIntoMatrix(Matrix output, int src, MPI mpi) throws MPIException {
        double[] tempBuffer = new double[output.getNRows()*output.getNCols()];
        mpi.MPI_Recv(tempBuffer,0,output.getNRows()*output.getNCols(),src,0,mpi.MPI_COMM_WORLD);
        for (int i = 0; i<tempBuffer.length; i++) {
            output.set(i/output.getNCols(),i%output.getNCols(),tempBuffer[i]);
        }
    }

    private static int getRowCount(int rank, int numRanks, int numRows) {
        int output = numRows/numRanks;
        if (rank < numRows%numRanks)
            output += 1;
        return output;
    }

    private static int getRowStart(int rank, int numRanks, int numRows) {
        int output = rank*numRows/numRanks;
        if (rank<numRows%numRanks)
            output += rank;
        else
            output += numRows%numRanks;
        return output;
    }
}
