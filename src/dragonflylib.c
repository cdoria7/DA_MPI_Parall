#include "dragonflylib.h"
#include <mpi.h>

/************* UTILS FUNCTIONS ************/

void *mycalloc(long dimension, int size, int id) {
    void *mem_loc = calloc(dimension, size);
    if (mem_loc == NULL) {
        fprintf(stderr, "[-] MALLOC ERROR, process %d \n", id);
        MPI_Abort(MPI_COMM_WORLD, MALLOC_ERROR);
    }
    return mem_loc;
}

void array_copy(double *dest, double *source, long dim) {
    memcpy(dest, source, sizeof(double) * dim);
}

void print_array_2d(double **array, long row, long col) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            fprintf(stdout, " %f ", array[i][j]);
        }
        fprintf(stdout, "\n");
    }
}

void print_array(double *array, long dim) {
    for (int i = 0; i < dim; i++) {
        fprintf(stdout, " %f ", array[i]);
    }
    fprintf(stdout, "\n");
}

void print_array_proc(double *array, long dim, int id) {
    fprintf(stdout, "Proc %d) ", id);
    for (int i = 0; i < dim; i++) {
        fprintf(stdout, " %f ", array[i]);
    }
    fprintf(stdout, "\n");
}

void print_array_onfile(double *array, long dim, FILE *fp) {
    for (int i = 0; i < dim; i++) {
        fprintf(fp, " %f ", array[i]);
    }
    fprintf(fp, "\n");
}

/****** ALGORITHM FUNCTION SUPPORT ********/

double update_radius(int ub, int lb, int iter, int max_iter) {
    return (ub - lb) / 4.0 + ((ub - lb) * ((double)iter / max_iter) * 2.0);
}

void init_struct_dragonfly(Dragonfly *dragonflies, int dragonfly_no, int dimension, int upperbound, int lowerbound, int rank, int seed) {
    srand48(seed + rank);
    for (int df = 0; df < dragonfly_no; df++) {
        dragonflies[df].position = (double *)calloc(dimension, sizeof(double));
        dragonflies[df].velocity = (double *)calloc(dimension, sizeof(double));
        dragonflies[df].fitness = 0.0;
        for (int i = 0; i < dimension; i++) {
            dragonflies[df].position[i] = drand48() * (upperbound - lowerbound) + lowerbound;
            dragonflies[df].velocity[i] = drand48() * (upperbound - lowerbound) + lowerbound;
        }
    }
    srand48(seed);
}

void init_neighbours(Neighbour *neighbours, int dimension) {
    neighbours = (Neighbour *)calloc(1, sizeof(Neighbour));
    neighbours[0].position = (double *)calloc(dimension, sizeof(double));
    neighbours[0].velocity = (double *)calloc(dimension, sizeof(double));
}

void add_neighbours(Neighbour **neighbours, int neighbour_no, int dimension) {
    (*neighbours) = (Neighbour *)realloc((*neighbours), (neighbour_no) * sizeof(Neighbour));
    (*neighbours)[neighbour_no - 1].position = (double *)calloc(dimension, sizeof(double));
    (*neighbours)[neighbour_no - 1].velocity = (double *)calloc(dimension, sizeof(double));
}

int check_bound(double *position, long upper_bound, long lower_bound, long dim) {
    for (int i = 0; i < dim; i++) {
        if (position[i] < upper_bound && position[i] > lower_bound) {
            return 1;
        }
    }
    return 0;
}

double *distance(double *x, double *y, long dim) {
    double *result = calloc(dim, sizeof(double));
    for (int i = 0; i < dim; i++)
        result[i] = sqrt(pow((y[i] - x[i]), 2.0));
    return result;
}

int validate_neighbour(double *dist, double radius, int dim) {
    for (int i = 0; i < dim; i++)
        if (dist[i] >= radius || dist[i] == 0)
            return 0;
    return 1;
}

int check_distance_radius(double *dist, double radius, int dim) {
    for (int i = 0; i < dim; i++)
        if (dist[i] >= radius)
            return 0;
    return 1;
}

int check_distance_is_zero(double *dist, double radius, int dim) {
    for (int i = 0; i < dim; i++)
        if (dist[i] == 0)
            return 0;
    return 1;
}

int food_near_dragonfly(double *dist, double radius, int dim) {
    for (int i = 0; i < dim; i++)
        if (dist[i] >= radius)
            return 1;
    return 0;
}

/************* BEHAVIOURS FUNCTIONS ************/

double *levy_func(long dim, int seed) {
    srand48(seed);
    double *levy = calloc(dim, sizeof(double));
    for (long i = 0; i < dim; i++) {
        double r1 = drand48() * Sigma;
        double r2 = fabs(drand48());
        levy[i] = 0.01 * (r1 / pow(r2, (1.0 / Beta)));
    }
    return levy;
}

void separation_dragonfly(double *separation, Dragonfly dragonflies, Neighbour *neighbours, int neighbour_no, long dim) {
    if (!neighbour_no)
        return;

    for (long i = 0; i < neighbour_no; i++)
        for (long k = 0; k < dim; k++)
            separation[k] += dragonflies.position[k] - neighbours[i].position[k];

    for (long k = 0; k < dim; k++)
        separation[k] = -separation[k];
}

void alignment_dragonfly(double *alignment, Neighbour *neighbours, long dim, int neighbour_no) {
    if (!neighbour_no)
        return;

    for (long i = 0; i < neighbour_no; i++)
        for (long j = 0; j < dim; j++)
            alignment[j] += neighbours[i].velocity[j];

    for (long i = 0; i < dim; i++)
        alignment[i] /= neighbour_no;
}

void cohesion_dragonfly(double *cohesion, Dragonfly dragonflies, Neighbour *neighbours, int neighbour_no, long dim) {
    double *cohesion_temp = (double *)calloc(dim, sizeof(double));

    if (neighbour_no > 1) {
        for (long i = 0; i < neighbour_no; i++)
            for (long j = 0; j < dim; j++)
                cohesion[j] += neighbours[i].position[j];

        for (long i = 0; i < dim; i++)
            cohesion[i] /= neighbour_no;
    } else {
        array_copy(cohesion_temp, dragonflies.position, dim);
    }

    for (long i = 0; i < dim; i++)
        cohesion[i] = cohesion_temp[i] - dragonflies.position[i];
}

void food_attraction_dragonfly(double *food_attraction, double *position, double *food_position, long dim) {
    for (long i = 0; i < dim; i++)
        food_attraction[i] = food_position[i] - position[i];
}

void predator_distraction_dragonfly(double *predator_distraction, double *position, double *predator_position, long dim) {
    for (long i = 0; i < dim; i++)
        predator_distraction[i] = predator_position[i] + position[i];
}

/************* TEST FUNCTIONS ************/
void *func_obj(int func, int *ub, int *lb) {
    switch (func) {
    case 1:
        *ub = 100;
        *lb = -100;
        return &TF1;
    case 2:
        *ub = 10;
        *lb = -10;
        return &TF2;

    case 3:
        *ub = 5;
        *lb = -5;
        return &ackley;

    case 5:
        *ub = 100;
        *lb = -100;
        return &TF5;

    default:
        *ub = 30;
        *lb = -30;
        return &TF5;
    }
}

double TF1(double *x, int dimension) {
    double result = 0.0;
    for (int i = 0; i < dimension; i++)
        result += pow(x[i], 2);
    return result;
}

double TF2(double *x, int dimension) {
    double result_sum = 0.0;
    double result_prod = 0.0;
    for (int i = 0; i < dimension; i++) {
        result_sum += fabs(x[i]);
        result_prod *= fabs(x[i]);
    }
    return result_prod + result_sum;
}

double TF5(double *x, int dimension) {
    double result = 0.0;
    for (int i = 0; i < dimension - 1; i++)
        result += 100.0 * (pow(x[i + 1] - pow(x[i], 2.0), 2.0)) + pow(x[i] - 1, 2.0);

    return result;
}

double ackley(double *x, double nDimensions) {
    double c = 2 * M_PI;
    double b = 0.2;
    double a = 20;
    double sum1 = 0;
    double sum2 = 0;
    int i;
    for (i = 0; i < nDimensions; i++) {
        sum1 = sum1 + pow(x[i], 2.0);
        sum2 = sum2 + cos(c * x[i]);
    }
    double term1 = -a * exp(-b * sqrt(sum1 / nDimensions));
    double term2 = -exp(sum2 / nDimensions);
    return term1 + term2 + a + M_E;
}
