//
//  main.cpp
//  CA-Lab2
//
//  Created by Zhenya Mordan on 9/16/19.
//  Copyright © 2019 Zhenya Mordan. All rights reserved.
//

#include "mpi.h"
#include <iostream>
#include <fstream>
using namespace std;

void read(const char* ,int*& , int& );
void write(const char* ,int* , int );
void merge(int *, int *, int, int, int);
void mergeSort(int *, int *, int, int);

int main(int argc, char** argv) {
    
    //init original array
    int n;
    int *original_array;
    //read array
    read(argv[1], original_array, n);
    
    //init mpi
    int world_rank;
    int world_size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    //divide the array in equal-sized chunks
    int size = n/world_size;
    
    //send each subarray to each process
    int *sub_array = new int [size];
    MPI_Scatter(original_array, size, MPI_INT, sub_array, size, MPI_INT, 0, MPI_COMM_WORLD);
    
    //mergesort on each process
    
    int *tmp_array = new int [size];
    mergeSort(sub_array, tmp_array, 0, (size - 1));
    
    //gather the sorted subarrays into one
    int *sorted = NULL;
    if(world_rank == 0) {
        
        sorted = new int [n];
        
    }
    
    MPI_Gather(sub_array, size, MPI_INT, sorted, size, MPI_INT, 0, MPI_COMM_WORLD);
    
    //make the final mergeSort call
    if(world_rank == 0) {
        
        int *other_array = new int [n];
        mergeSort(sorted, other_array, 0, (n - 1));
        
        //write sorted array
        write(argv[2], sorted, n);
        
        //clean up root
        delete [] sorted;
        delete [] other_array;
        
    }
    
    //clean up rest
    delete [] original_array;
    delete [] sub_array;
    delete [] tmp_array;
    
    //finalize MPI
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    
    //print runtime
    if(world_rank==0){
        cout << "Total time = " << clock()/1000.0 << endl;
    }
}

void write(const char* filename,int* array, int n){
    ofstream ofile(filename);
    
    cout<< "Writting array...\n";
    if(ofile.is_open()){
        for(int i=0;i<n;){
            ofile<<array[i++]<<"\n";
        }
    }
    ofile.close();
}
void read(const char* filename,int*& array, int &n){
    
    ifstream ifile(filename);
    
    ifile>>n;
    
    array = new int [n];
    
    if(ifile.is_open()){
        for(int i=0;i<n;i++){
            ifile>>array[i];
        }
    }
    ifile.close();
}

void merge(int *a, int *b, int l, int m, int r) {
    
    int h, i, j, k;
    h = l;
    i = l;
    j = m + 1;
    
    while((h <= m) && (j <= r)) {
        if(a[h] <= a[j]) {
            
            b[i] = a[h];
            h++;
            
        }
        else {
            
            b[i] = a[j];
            j++;
        }
        i++;
    }
    if(m < h) {
        for(k = j; k <= r; k++) {
            b[i] = a[k];
            i++;
        }
    }
    else {
        for(k = h; k <= m; k++) {
            b[i] = a[k];
            i++;
        }
    }
    
    for(k = l; k <= r; k++) {
        
        a[k] = b[k];
        
    }
    
}

void mergeSort(int *a, int *b, int l, int r) {
    
    int m;
    
    if(l < r) {
        
        m = (l + r)/2;
        
        mergeSort(a, b, l, m);
        mergeSort(a, b, (m + 1), r);
        merge(a, b, l, m, r);
        
    }
    
}
