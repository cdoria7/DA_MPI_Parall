/**************** LIBRARY ****************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/**************** CONSTANTS ****************/

#define Beta 1.5
#define Sigma 0.69658
#define _Epsilon 0.00000000000001

#define OPEN_FILE_ERROR -1
#define MALLOC_ERROR -2
#define TYPE_ERROR -3
#define ARGUMENTS_ERROR -4
#define GENERIC_ERROR -99

#define TRUE 1
#define FALSE 0

#define BLOCK_LOW(id, p, n) ((id) * (n) / (p))
#define BLOCK_HIGH(id, p, n) (BLOCK_LOW((id) + 1, p, n) - 1)
#define BLOCK_SIZE(id, p, n) (BLOCK_HIGH(id, p, n) - BLOCK_LOW(id, p, n) + 1)

/**************** STRUCT ****************/

typedef struct Neighbours_t {
    double *position;
    double *velocity;
} Neighbour;

typedef struct Dragonfly_t {
    double *position;
    double *velocity;
    double fitness;
    Neighbour *neighbours;
    int neighbour_no;
} Dragonfly;

typedef struct result_t {
    double best_score;
    double *best_position;
} Result;

/************* UTILS FUNCTIONS ************/

void print_array_2d(double **matrix, long row, long col);
void print_array(double *array, long dim);
void print_array_proc(double *array, long dim, int id);
void print_array_onfile(double *array, long dim, FILE *fp);
void array_copy(double *dest, double *source, long dim);
void *mycalloc(long dimension, int size, int id);

/****** ALGORITHM FUNCTION SUPPORT ********/

void init_struct_dragonfly(Dragonfly *dragonflies, int dragonfly_no, int dimension, int upperbound, int lowerbound, int rank, int seed);
void add_neighbours(Neighbour **neighbour, int neighbours_no, int dimension);
double update_radius(int ub, int lb, int iter, int max_iter);
void update_weight(int iter, int max_iter, double *w, double *s, double *a, double *c, double *f, double *e);
double *distance(double *x, double *y, long dim);
int check_bound(double *position, long upper_bound, long lower_bound, long dim);
int check_distance_radius(double *dist, double radius, int dim);
int food_near_dragonfly(double *dist, double radius, int dim);
int check_distance_zero(double *dist, double radius, int dim);
int validate_neighbour(double *dist, double radius, int dim);

/************* BEHAVIOURS FUNCTIONS ************/
double *levy_func(long dim, int seed);
void separation_dragonfly(double *separation, Dragonfly dragonflies, long dim);
void alignment_dragonfly(double *alignment, Neighbour *neighbours, long dim, long neighbours_no);
void cohesion_dragonfly(double *cohesion, Dragonfly dragonflies, long dim);
void food_attraction_dragonfly(double *food_attraction, double *position, double *food_position, long dim);
void predator_distraction_dragonfly(double *predator_distraction, double *position, double *predator_position, long dim);

/************* TEST FUNCTIONS ************/

#define SPHERE 1
#define ROSENBROCK 5

void *func_obj(int func, int *ub, int *lb);
double TF1(double *x, int dimension);
double TF2(double *x, int dimension);
double TF5(double *x, int dimension);
double ackley(double *x, double nDimensions);