//
//  main.cpp
//  CA-Lab3
//
//  Created by Zhenya Mordan on 11/7/19.
//  Copyright © 2019 Zhenya Mordan. All rights reserved.
//
//

#include <iomanip>
#include <iostream>
#include <string>
#include <ctime>
#include <math.h>
#include <fstream>
#include "mpi.h"

#include <stdio.h>
#include <stdlib.h>

void allocateMemMatrix(float**& A, int n);
void allocateMemMatrix(float**& A, int m, int n);
void fillMatrix(float** A, int n);
void fillMatrixAs1(float** A, int n);
void writeMatrix(const char* fileName, const char* name, float** A, int n);
void writeMatrix(const char* fileName, const char* name, float** A, int m, int n);
void solveWithGaussMethod(float **A, float **X, float **B, int n);
void directMove(float **A, float **B, int n);
void reverseMove(float **A, float **X, float **B, int n, int a, int b);

using namespace std;
int main(int argc, char** argv) {
    
    int n=atoi(argv[1]); //matrix size
    
    float **B;
    float **A;
    float **C;
    float **X;
    
    //prepare matrix B
    allocateMemMatrix(B, n);
    fillMatrixAs1(B, n);
    
    //prepare matrix A
    allocateMemMatrix(A, n);
    fillMatrix(A, n);
    
    //init mpi
    int worldRank;
    int worldSize;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Status status;
    
    int size = n/worldSize;
    
    //direct move
    if(worldRank==0){
        writeMatrix("matrix_A","A", A, n);
        writeMatrix("matrix_B","B", B, n);
        directMove(A, B, n);
        writeMatrix("matrix_A_after_direct_move","A after direct move", A, n);
        writeMatrix("matrix_B_after_direct_move","B after direct move", B, n);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for(int i=0;i<n;i++){
        MPI_Bcast(A[i], n, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    for(int i=0;i<n;i++){
        MPI_Bcast(B[i], n, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    //reverse move
    //prepare matrix X
    allocateMemMatrix(X, n,size);
    //reverse move
    reverseMove(A, X, B, n, worldRank*size, (worldRank+1)*size);
    MPI_Barrier(MPI_COMM_WORLD);
    //prepare matrix С
    allocateMemMatrix(C, n);
    for(int i=0;i<n;i++){
        MPI_Gather(X[i], size, MPI_FLOAT, C[i], size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    //finalize MPI
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    //print runtime
    if(worldRank==0){
        cout << "Total time = " << clock()/1000.0 << endl;
        writeMatrix("matrix_C", "C", C, n);
    }
    return 0;
}
void allocateMemMatrix(float**& A, int n)
{
    A = new float*[n]; //i
    for (int j = 0; j < n; j++)
    {
        
        A[j] = new float[n]; //j
    }
}
void allocateMemMatrix(float**& A, int m, int n)
{
    A = new float*[m]; //i
    for (int j = 0; j < m; j++)
    {
        A[j] = new float[n]; //j
    }
}
void fillMatrix(float** A, int n)
{
    srand(time(0));
    for (int i = 0; i < n; i++) //set whole matrix
    {
        for (int j = 0; j < n; j++)
        {
            A[i][j] = (rand() % 9);
        }
    }
    for (int i = 0; i < n; i++) //set diagonal elements aii
    {
        A[i][i] = 0;
        for (int j = 0; j < n; j++)
        {
            if (i != j)
            {
                A[i][i] += A[i][j];
            }
        }
    }
}
void fillMatrixAs1(float** A, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if(i==j){
                A[i][j] = 1;
            }
            else{
                A[i][j] = 0;
            }
        }
    }
}
void writeMatrix(const char* filename, const char* name, float** A, int n)
{
    ofstream ofile(filename);
    cout<< "Writting matrix "<< name <<"\n";
    if(ofile.is_open()){
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < n; i++)
            {
                ofile << fixed << setprecision(8)<< A[j][i] << "\t\t";
            }
            ofile << endl;
        }
    }
    else
    {
        cout<<"Can not write matrix"<< name << " to the file "<< filename<<"\n";
    }
    ofile.close();
}
void writeMatrix(const char* filename, const char* name, float** A, int m, int n)
{
    ofstream ofile(filename);
    cout<< "Writting matrix "<< name <<"\n";
    if(ofile.is_open()){
        for (int j = 0; j < m; j++)
        {
            for (int i = 0; i < n; i++)
            {
                ofile << fixed << setprecision(8)<< A[j][i] << "\t\t";
            }
            ofile << endl;
        }
    }
    else
    {
        cout<<"Can not write matrix"<< name << " to the file "<< filename<<"\n";
    }
    ofile.close();
}

void directMove(float **A, float **B, int n)
{
    cout<<"Direct move...\n";
    for (int k = 0; k < n - 1; k++) // k = 1, 2, … , n–1:
    {
        for (int i = k + 1; i < n; i++) //i = k + 1, k + 2, …, n:
        {
            float l = A[i][k] / A[k][k]; // temp l[i][k]= A[i][k]/A[k][k]
            
            for(int p=0;p<n; p++){
                B[i][p] = B[i][p] - l * B[k][p];
            }
            for (int j = k; j < n; j++) //j = k + 1, k + 2, …, n:
            {
                A[i][j] = A[i][j] - l * A[k][j];
            }
        }
    }
}
void reverseMove(float **A, float **X, float **B, int n,int a, int b)
{
    cout<<"Reverse move...\n";
    for(int p=0;p<b-a;p++){
        X[n - 1][p] = B[n - 1][p+a] / A[n - 1][n - 1];
        for (int i = n - 2; i > -1; i--) //i=n–1, n–2,...,1.
        {
            float l = 0; //l = sum(A [i][j]X[j]) below
            for (int j = i + 1; j < n; j++) //for j=i+1 to n
            {
                l += A[i][j] * X[j][p];
            }
            X[i][p] = (B[i][p+a] - l) / A[i][i];
        }
    }
}


