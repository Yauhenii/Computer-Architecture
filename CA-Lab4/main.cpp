//  main.cpp
//  CA-Lab4
//
//  Created by Zhenya Mordan on 12/15/19.
//  Copyright © 2019 Zhenya Mordan. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include "mpi.h"
#include <cmath>

// 1D composition

using namespace std;

ofstream fout("output.txt");

int n = 100; // points count for y
double lower_border_y = 0, upper_border_y = 1; //

int m = 100; // points count for x
double lower_border_x = 0, upper_border_x = 1;

double hx = (upper_border_x - lower_border_x) / (m - 1); // step by x
double hy = (upper_border_y - lower_border_y) / (n - 1); // step by y

double yc = 1 / (hy * hy); //  1 / (hy * hy)
double xc = 1 / (hx * hx); // 1 / (hx * hx)
double c = 2 * (xc + yc); // 2 * (1 / (hx * hx) + 1 / (hy * hy))


double f(double x, double y)
{
    return x*y;
}

double f1(double y)
{
    return sin(y);
}

double f2(double y)
{
    return y;
}

double f3(double x)
{
    return x*x*x;
}

double f4(double x)
{
    return x*x;
}

void fill(int* recvcounts, int* displs, int size, int l);

int main(int argc, char ** argv)
{
    int rank, size;
    MPI_Status status;

    double time_start;
    double* u = new double[n * m];
    double* u_result = new double[n * m]; // result value table 

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // fill values of table u by zeros
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
            u[i * m + j] = 0;
    }

    int l = (n - 2) / size + 2; // rows count per process

    int first = rank * (l - 2); // [first index,
    int last = first + l; // ,last index)

    double x; // current x
    double y; // current y

    // fill border values
    // for bottom and upper border (only in main process and last process)
    if (rank == 0)
    {
        x = lower_border_x;
        for (int j = 0; j < m; j++)
        {
            u[0 * m + j] = f3(x);
            x += hx;
        }
    }

    if (rank == size - 1)
    {
        x = lower_border_x;
        for (int j = 0; j < m; j++)
        {
            u[(n - 1) * m + j] = f4(x);
            x += hx;
        }
    }

    y = lower_border_y + first * hy;

    // for left and right borders (in all processes)
    for (int i = first; i < last; i++)
    {
        u[i * m] = f1(y);
        u[i * m + m - 1] = f2(y);
        y += hy;
    }

    // start of time
    if (rank == 0)
        time_start = MPI_Wtime();

    // begin of computation
    double epsilon = 1e-6; // wanted accuracy
    double error = 1; // current accuracy

    while (error > epsilon)
    {
        error = 0;
        // difference schema for numerical solving of differential equation
        y = lower_border_y + (first + 1) * hy;
        for (int i = first + 1; i < last - 1; i++)
        {
            x = lower_border_x + hx;
            for (int j = 1; j < m - 1; j++)
            {
                double uprev = u[i * m + j];
                u[i * m + j] = (-f(x, y) + yc * (u[(i - 1) * m + j] + u[(i + 1) * m + j]) + xc * (u[i * m + j - 1] + u[i * m + j + 1])) / c;
                error = max(error, abs(uprev - u[i * m + j]));
                x += hx;
            }
            y += hy;
        }

        // get error from all additional processes in main + calculate max error
        if (rank == 0)
        {
            double currentError;
            for (int i = 1; i < size; i++)
            {
                MPI_Recv(&currentError, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
                error = max(error, currentError);
            }
        }
        else
            MPI_Send(&error, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

        // send max delta from main process to all additional
        if (rank == 0)
            for (int i = 1; i < size; i++)
                MPI_Send(&error, 1, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
        else
            MPI_Recv(&error, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);

        // send new rows to neighbours
        if (rank < size - 1)
        {
            MPI_Send(&u[(last - 2) * m], m, MPI_DOUBLE, rank + 1, 2, MPI_COMM_WORLD);
            MPI_Recv(&u[(last - 1) * m], m, MPI_DOUBLE, rank + 1, 2, MPI_COMM_WORLD, &status);
        }

        if (rank > 0)
        {
            MPI_Send(&u[(first + 1) * m], m, MPI_DOUBLE, rank - 1, 2, MPI_COMM_WORLD);
            MPI_Recv(&u[first * m], m, MPI_DOUBLE, rank - 1, 2, MPI_COMM_WORLD, &status);
        }
    }

    // fill recvcounts (count of receved process data) and displs (positions of sent proccess data)
    int* recvcounts = new int[size], *displs = new int[size];
    if (rank == 0)
    {
        fill(recvcounts, displs, size, l);
    }
    
    MPI_Gatherv(&u[(first + 2 * (rank > 0)) * m], (l - 2 * (rank > 0)) * m, MPI_DOUBLE, u_result, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // output computation time
    if (rank == 0)
        cout << MPI_Wtime() - time_start << endl;

    // write answer to file
    if (rank == 0)
    {
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
                fout << fixed << setprecision(6) << lower_border_x+j*hx <<" "<<lower_border_y+i*hy <<" "<< u_result[i * m + j] << "\n";
        }
    }

    MPI_Finalize();

    return 0;
}

void fill(int* recvcounts, int* displs, int size, int l)
{
    recvcounts[0] = l * m;
    for (int i = 1; i < size; i++)
        recvcounts[i] = (l - 2) * m;

    displs[0] = 0;
    for (int i = 1; i < size; i++)
        displs[i] = ((l - 2) * i + 2) * m;
}
