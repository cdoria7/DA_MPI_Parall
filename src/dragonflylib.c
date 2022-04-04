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

void print_array(double *array, long dim) {
    for (int i = 0; i < dim; i++) {
        fprintf(stdout, " %f ", array[i]);
    }
    fprintf(stdout, "\n");
}

/*
void print_array_2d(double **array, long row, long col) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            fprintf(stdout, " %f ", array[i][j]);
        }
        fprintf(stdout, "\n");
    }
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
*/

/****** ALGORITHM FUNCTION SUPPORT ********/

/**
 * @brief Inizializza l'array di strutture di tipo Dragonfly e assegna un posizione
 *        pseudo-randomica ad ognuna di esse basandosi su upperbound e lowerbound.
 *
 * @param dragonflies Array di Strutture Dragonfly
 * @param dragonfly_no dimensione dell'array
 * @param dim dimensione del problema (numero di variabili del problema)
 * @param upperbound Upperbound dello spazio di ricerca
 * @param lowerbound Lowerbound dello spazio di ricerca
 * @param rank Rank del processo
 * @param seed Seed per la generazione pseudo-randomica
 */
void init_struct_dragonfly(Dragonfly *dragonflies, int dragonfly_no, int dim, int upperbound, int lowerbound, int rank, int seed) {
    srand48(seed + rank);
    for (int df = 0; df < dragonfly_no; df++) {
        dragonflies[df].position = (double *)calloc(dim, sizeof(double));
        dragonflies[df].velocity = (double *)calloc(dim, sizeof(double));
        dragonflies[df].fitness = 0.0;
        for (int i = 0; i < dim; i++) {
            dragonflies[df].position[i] = drand48() * (upperbound - lowerbound) + lowerbound;
            dragonflies[df].velocity[i] = drand48() * (upperbound - lowerbound) + lowerbound;
        }
    }
    srand48(seed);
}

/**
 * @brief Inizializza l'array di strutture che conterrà le libellule vicine
 *
 * @param neighbours Array di strutture dei vicini.
 * @param dim dimensione del problema (numero di variabili del problema)
 */
void init_neighbours(Neighbour *neighbours, int dim) {
    neighbours = (Neighbour *)calloc(1, sizeof(Neighbour));
    neighbours[0].position = (double *)calloc(dim, sizeof(double));
    neighbours[0].velocity = (double *)calloc(dim, sizeof(double));
}

/**
 * @brief Libera lo spazio in memoria occupato dall'array dei vicini
 *
 * @param neighbours Array di strutture dei vicini.
 * @param neighbour_no numero degli elementi nell'array dei vicini
 */
void freeNeighbours(Neighbour *neighbours, int neighbour_no) {
    for (int n = 0; n < neighbour_no; n++) {
        free(neighbours[n].position), neighbours[n].position = NULL;
        free(neighbours[n].velocity), neighbours[n].velocity = NULL;
    }
}

/**
 * @brief Aggiunge un elemento di tipo Neighbour al vettore dei vicini.
 *
 * @param neighbours Array di strutture dei vicini.
 * @param neighbour_no numero degli elementi nell'array dei vicini
 * @param dim dimensione del problema (numero di variabili del problema)
 */
void add_neighbours(Neighbour **neighbours, int neighbour_no, int dim) {
    (*neighbours) = (Neighbour *)realloc((*neighbours), (neighbour_no) * sizeof(Neighbour));
    (*neighbours)[neighbour_no - 1].position = (double *)calloc(dim, sizeof(double));
    (*neighbours)[neighbour_no - 1].velocity = (double *)calloc(dim, sizeof(double));
}

/**
 * @brief
 *
 * @param position   Posizione della libellula da controllare
 * @param upperbound Upperbound dello spazio di ricerca
 * @param lowerbound Lowerbound dello spazio di ricerca
 * @param dim  Dimensione del problema (numero di variabili del problema)
 * @return int       Ritorna 0 se la posizione è all'interno dello spazio di ricerca
 */
int check_bound(double *position, long upper_bound, long lower_bound, long dim) {
    for (int i = 0; i < dim; i++) {
        if (position[i] < upper_bound && position[i] > lower_bound) {
            return 1;
        }
    }
    return 0;
}

/**
 * @brief
 *
 * @param x posizione del primo punto
 * @param y posizione del secondo punto
 * @param dim  Dimensione del problema (numero di variabili del problema)
 * @return double* Ritorna un vettore delle distanze di ogni componente
 */
double *distance(double *x, double *y, long dim) {
    double *result = calloc(dim, sizeof(double));
    for (int i = 0; i < dim; i++)
        result[i] = sqrt(pow((y[i] - x[i]), 2.0));
    return result;
}

/**
 * @brief Verifica che il candidato vicino possa essere considerato realmente un vicinp
 *
 * @param dist distanza della libellula del candidato vicino
 * @param radius raggio della libellula
 * @param dim  Dimensione del problema (numero di variabili del problema)
 * @return int Ritorna 0 se il vicino è fuori dal raggio o se tutte le distanze sono nulle,
 *             altrimenti 1 se il vicino candidato è un vicino effettivo.
 */
int validate_neighbour(double *dist, double radius, int dim) {
    int flagzero = 0;

    for (int i = 0; i < dim; i++) {
        if (dist[i] >= radius)
            return 0;

        if (dist[i] == 0)
            flagzero += 1;
    }
    if (flagzero == dim)
        return 0;

    return 1;
}

/**
 * @brief Verifica che una distanza sia inferiore del raggio.
 *
 * @param dist distanza della libellula del candidato vicino
 * @param radius raggio della libellula
 * @param dim  Dimensione del problema (numero di variabili del problema)
 * @return int Ritorna 1 se la distanza è minore del raggio della libellula
 */
int check_distance_radius(double *dist, double radius, int dim) {
    for (int i = 0; i < dim; i++)
        if (dist[i] >= radius)
            return 0;
    return 1;
}

/**
 * @brief Verifica che una distanza sia inferiore del raggio.
 *
 * @param dist distanza della libellula del candidato vicino
 * @param radius raggio della libellula
 * @param dim  Dimensione del problema (numero di variabili del problema)
 * @return int Ritorna 1 se il cibo si trova all'interno del raggio della libellua
 */
int food_near_dragonfly(double *dist, double radius, int dim) {
    for (int i = 0; i < dim; i++)
        if (dist[i] >= radius)
            return 1;
    return 0;
}

/************* SWARM FUNCTIONS ************/

/**
 * @brief Funzione che descrive il moto di separazione delle libellule Eq.1 del report
 *
 * @param separation vettore che contiene le componenti del moto di separazione
 * @param dragonfly libellula sulla quale calcolare il moto di separazione
 * @param neighbours vicini della libellula sulla quale calcolare il moto di separazione
 * @param neighbour_no numero di libellule vicine alla libellula ulla quale calcolare il moto di separazione
 * @param dim Dimensione del problema (numero di variabili del problema)
 */
void separation_dragonfly(double *separation, Dragonfly dragonfly, Neighbour *neighbours, int neighbour_no, long dim) {
    if (!neighbour_no)
        return;

    for (long i = 0; i < neighbour_no; i++)
        for (long k = 0; k < dim; k++)
            separation[k] += dragonfly.position[k] - neighbours[i].position[k];

    for (long k = 0; k < dim; k++)
        separation[k] = -separation[k];
}

/**
 * @brief Funzione che descrive il moto di allineamento delle libellule Eq.2 del report
 *
 * @param alignment vettore che contiene le componenti del moto di allineamento
 * @param neighbours vicini della libellula sulla quale calcolare il moto di separazione
 * @param neighbour_no numero di libellule vicine alla libellula ulla quale calcolare il moto di separazione
 * @param dim Dimensione del problema (numero di variabili del problema)
 */
void alignment_dragonfly(double *alignment, Neighbour *neighbours, int neighbour_no, long dim) {
    if (!neighbour_no)
        return;

    for (long i = 0; i < neighbour_no; i++)
        for (long j = 0; j < dim; j++)
            alignment[j] += neighbours[i].velocity[j];

    for (long i = 0; i < dim; i++)
        alignment[i] /= neighbour_no;
}

/**
 * @brief Funzione che descrive il moto di coesione delle libellule Eq.3 del report
 *
 * @param cohesion vettore che contiene le componenti del moto di coesione
 * @param dragonfly libellula sulla quale calcolare il moto di separazione
 * @param neighbours vicini della libellula sulla quale calcolare il moto di separazione
 * @param neighbour_no numero di libellule vicine alla libellula ulla quale calcolare il moto di separazione
 * @param dim Dimensione del problema (numero di variabili del problema)
 */
void cohesion_dragonfly(double *cohesion, Dragonfly dragonfly, Neighbour *neighbours, int neighbour_no, long dim) {
    double *cohesion_temp = (double *)calloc(dim, sizeof(double));

    if (neighbour_no > 1) {
        for (long i = 0; i < neighbour_no; i++)
            for (long j = 0; j < dim; j++)
                cohesion[j] += neighbours[i].position[j];

        for (long i = 0; i < dim; i++)
            cohesion[i] /= neighbour_no;
    } else {
        memcpy(cohesion_temp, dragonfly.position, dim * sizeof(double));
    }

    for (long i = 0; i < dim; i++)
        cohesion[i] = cohesion_temp[i] - dragonfly.position[i];
}

/**
 * @brief Funzione che descrive il moto di attrazione verso il cibo delle libellule Eq.4 del report
 *
 * @param food_attraction vettore che contiene le componenti del moto di attrazione verso il cibo
 * @param position Posizione della libellula
 * @param food_position Posizione del cibo
 * @param dim Dimensione del problema (numero di variabili del problema)
 */
void food_attraction_dragonfly(double *food_attraction, double *position, double *food_position, long dim) {
    for (long i = 0; i < dim; i++)
        food_attraction[i] = food_position[i] - position[i];
}

/**
 * @brief Funzione che descrive il moto di allontanamento dal predatore delle libellule Eq.5 del report
 *
 * @param predator_distraction vettore che contiene le componenti del moto di allontanamento dal predatore
 * @param position Posizione della libellula
 * @param food_position Posizione del cibo
 * @param dim Dimensione del problema (numero di variabili del problema)
 */
void predator_distraction_dragonfly(double *predator_distraction, double *position, double *predator_position, long dim) {
    for (long i = 0; i < dim; i++)
        predator_distraction[i] = predator_position[i] + position[i];
}

/**
 * @brief Funzione di Levy che descrive un movimento pseudorandomico all'interno dello spazio di ricerca.
 *
 * @param dim Dimensione del problema (numero di variabili del problema)
 * @param seed Seed per la generazione pseudo-randomica
 * @return double*  vettore che contiene le componenti del moto di allontanamento dal predatore
 */
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

/************* TEST FUNCTIONS ************/

/**
 * @brief Restituisce una funzione di test.
 *
 * @param func valore intero associato alla funzione:
 *              1: Sphere function
 *              2: (TF2) Somma tra la sommatoria e la produttoria delle componenti
 *              3: Rosenbrock function
 *              other: Viene ritornata Rosembrock
 *
 * @param ub Upperbound dello spazio di ricerca
 * @param lb Lowerbound dello spazio di ricerca
 * @return void* Ritorna un puntatore a funzione
 */
void *func_obj(int func, int *ub, int *lb) {
    switch (func) {
    case 1:
        *ub = 100;
        *lb = -100;
        return &Sphere;

    case 2:
        *ub = 10;
        *lb = -10;
        return &TF2;

    case 3:
        *ub = 30;
        *lb = -30;
        return &Rosenbrock;

    default:
        *ub = 30;
        *lb = -30;
        return &Rosenbrock;
    }
}

char *obj_function_name(int func) {
    switch (func) {
    case 1:
        return "Sphere";
    case 2:
        return "TF2";
    case 3:
        return "Rosenbrock";
    default:
        return "Rosenbrock";
    }
}

double Sphere(double *x, int dimension) {
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

double Rosenbrock(double *x, int dimension) {
    double result = 0.0;
    for (int i = 0; i < dimension - 1; i++)
        result += 100.0 * (pow(x[i + 1] - pow(x[i], 2.0), 2.0)) + pow(x[i] - 1, 2.0);

    return result;
}
