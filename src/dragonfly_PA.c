#include "dragonflylib.h"
#include <mpi.h>
#include <unistd.h>

#define DYNAMIC_WEIGHT 1

typedef struct hotspot_t {
    double value;
    int id;
} HotSpot;

void DA_Parallel(int id, int p, int iteration, int dragonfly_no, int dim, int test_func, Result result);

int main(int argc, char *argv[]) {
    int id, p;     // 8 byte
    Result result; // 34 byte

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    /* Arguments Check */
    if (!id) {
        if (argc < 5) {
            fprintf(stderr, "Argument Error!\n");
            fprintf(stderr, "mpirun -n [processors] Usage: %s [iter] [dragonfly_no] [dim] [test_func] [<repeat>]", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, ARGUMENTS_ERROR);
        }
    }

    int repeat = argc == 6 ? atoi(argv[5]) : 1; // 4 byte

    int iteration = atoi(argv[1]);    // 4 byte
    int dragonfly_no = atoi(argv[2]); // 4 byte
    int dim = atoi(argv[3]);          // 4 byte
    int test_func = atoi(argv[4]);    // 4 byte

    result.best_position = (double *)mycalloc(dim, sizeof(double), id);

    for (int r = 0; r < repeat; r++)
        DA_Parallel(id, p, iteration, dragonfly_no, dim, test_func, result);

    MPI_Finalize();
    return 0;
}

void DA_Parallel(int id, int p, int iteration, int dragonfly_no, int dim, int test_func, Result result) {

    int seed = 72024; // 4 byte

    srand48(seed);

    double elapsed_time = 0.0; // 8 byte
    double global_time = 0.0;  // 8 byte

    int upper_bound = 0; // 4 byte
    int lower_bound = 0; // 4 byte

    /* Test Function */
    double (*fitness_function)(double *x, int dimension);
    fitness_function = func_obj(test_func, &upper_bound, &lower_bound);

    /* Init radius and max velocity */
    double radius = (upper_bound - lower_bound) / 20.0;       // 8 byte
    double velocity_max = (upper_bound - lower_bound) / 20.0; // 8 byte

    /* Dragonflies for processor */
    int dragonfly_proc_no = BLOCK_SIZE(id, p, dragonfly_no);                                  // 4 byte
    Dragonfly *dragonflies = (Dragonfly *)mycalloc(dragonfly_proc_no, sizeof(Dragonfly), id); // 56 * 3000 = 168.000 byte
    init_struct_dragonfly(dragonflies, dragonfly_proc_no, dim, upper_bound, lower_bound, id, seed);

    int neighbour_no = 0;                                                 // 4 byte
    int *neighbours_index = NULL;                                         // 4 byte
    Neighbour *neighbours = calloc(dragonfly_proc_no, sizeof(Neighbour)); //  48 * 3000 = 144.000 byte
    for (int i = 0; i < dragonfly_proc_no; i++) {
        neighbours[i].position = (double *)calloc(dim, sizeof(double));
        neighbours[i].velocity = (double *)calloc(dim, sizeof(double));
    }

    /*********************** Global Variable ***********************/
    HotSpot food_value, pred_value; // 12 + 12 = 24 byte

    /** Food & Predator Value **/
    food_value.id = -1;
    food_value.value = INFINITY;

    pred_value.id = -1;
    pred_value.value = -INFINITY;

    /** Food & Predator Position **/
    double *food_pos = (double *)mycalloc(dim, sizeof(double), id); // 3 * 8 = 24 byte
    double *pred_pos = (double *)mycalloc(dim, sizeof(double), id); // 3 * 8 = 24 byte

    double *local_food_pos = (double *)mycalloc(dim, sizeof(double), id); // 3 * 8 = 24 byte
    double *local_pred_pos = (double *)mycalloc(dim, sizeof(double), id); // 3 * 8 = 24 byte

    /*********************** Local Variable ***********************/
    HotSpot local_food_value, local_pred_value; // 12 + 12 = 24 byte

    /** Food & Predator Value **/
    local_food_value.id = id;
    local_food_value.value = INFINITY;

    local_pred_value.id = id;
    local_pred_value.value = -INFINITY;

    /** Food & Predator Distance **/
    double *food_distance = (double *)mycalloc(dim, sizeof(double), id); // 3 * 8 = 24 byte
    double *pred_distance = (double *)mycalloc(dim, sizeof(double), id); // 3 * 8 = 24 byte

    /** Distance between dragonflies **/
    double *dist = (double *)mycalloc(dim, sizeof(double), id); // 3 * 8 = 24 byte

    /** Levy Vector **/
    double *levy = (double *)mycalloc(dim, sizeof(double), id); // 3 * 8 = 24 byte

    /** Behaviours Vector **/
    double *separation = (double *)mycalloc(dim, sizeof(double), id);       // 3 * 8 = 24 byte
    double *alignment = (double *)mycalloc(dim, sizeof(double), id);        // 3 * 8 = 24 byte
    double *cohesion = (double *)mycalloc(dim, sizeof(double), id);         // 3 * 8 = 24 byte
    double *food_attraction = (double *)mycalloc(dim, sizeof(double), id);  // 3 * 8 = 24 byte
    double *pred_distraction = (double *)mycalloc(dim, sizeof(double), id); // 3 * 8 = 24 byte

    double *flagUb = (double *)calloc(dim, sizeof(double));
    double *flagLb = (double *)calloc(dim, sizeof(double));

    /** Define Weight **/
#if DYNAMIC_WEIGHT
    // 7 * 8 = 56 byte
    double w = 0.0;
    double s = 0.0;
    double a = 0.0;
    double c = 0.0;
    double f = 0.0;
    double e = 0.0;
    double my_c = 0.0;
#endif

#if !DYNAMIC_WEIGHT
    double w = drand48() * (0.9 - 0.2) + 0.2;
    double s = 0.1;
    double a = 0.1;
    double c = 0.7;
    double f = 1;
    double e = 1;
#endif

    // /** MPI_Datatype User-Defined **/
    // MPI_Datatype MPI_POSITION_TYPE;
    // MPI_Type_contiguous(dim, MPI_DOUBLE, &MPI_POSITION_TYPE);
    // MPI_Type_commit(&MPI_POSITION_TYPE);

    // /** Non-Blocking AllReduce: Request & Status **/
    // MPI_Request req;
    // MPI_Status st;

    while (1)
        ;

    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time = -MPI_Wtime();

    /** Start DA Algorithm **/
    for (int iter = 0; iter <= iteration; iter++) {

        /** Update Food & Pred Position and Value **/
        for (int df = 0; df < dragonfly_proc_no; df++) {

            /** Compute fitness function **/
            dragonflies[df].fitness = fitness_function(dragonflies[df].position, dim);

            /** Find Food Fitness  **/
            if (dragonflies[df].fitness < food_value.value && dragonflies[df].fitness < local_food_value.value) {
                local_food_value.value = dragonflies[df].fitness;
                memcpy(food_pos, dragonflies[df].position, dim * sizeof(double));
            }

            /** Find Pred Fitness  **/
            if (dragonflies[df].fitness > pred_value.value && dragonflies[df].fitness > local_pred_value.value) {
                local_pred_value.value = dragonflies[df].fitness;
                memcpy(pred_pos, dragonflies[df].position, dim * sizeof(double));
            }
        }

        /** Find global food value and global predator value **/
        MPI_Allreduce(&local_food_value, &food_value, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
        MPI_Allreduce(&local_pred_value, &pred_value, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

        /** Broadcasting the Global Food and Global Pred position**/
        MPI_Bcast(food_pos, dim, MPI_DOUBLE, food_value.id, MPI_COMM_WORLD);
        MPI_Bcast(pred_pos, dim, MPI_DOUBLE, pred_value.id, MPI_COMM_WORLD);

        /** Add neighbours **/
        for (int df = 0; df < dragonfly_proc_no; df++) {

            // Trovo gli indici dei vicini
            for (int i = 0; i < dragonfly_proc_no; i++) {

                distance(dist, dragonflies[df].position, dragonflies[i].position, dim);

                if (validate_neighbour(dist, radius, dim)) {
                    neighbour_no += 1;
                    memcpy(neighbours[df].position, dragonflies[i].position, dim);
                    memcpy(neighbours[df].velocity, dragonflies[i].velocity, dim);
                    // neighbours_index = realloc(neighbours_index, neighbour_no * sizeof(int));
                    // neighbours_index[neighbour_no - 1] = i;
                }
            }

            // // Istanzio la struttura che contiene i vicini
            // neighbours = (Neighbour *)mycalloc(neighbour_no, sizeof(Neighbour), id);
            // for (int i = 0; i < neighbour_no; i++) {
            //     neighbours[i].position = calloc(dim, sizeof(double));
            //     neighbours[i].velocity = calloc(dim, sizeof(double));
            // }

            // for (int i = 0; i < neighbour_no; i++) {
            //     memcpy(neighbours[0].position, dragonflies[neighbours_index[i]].position, dim);
            //     memcpy(neighbours[0].velocity, dragonflies[neighbours_index[i]].velocity, dim);
            // }

            // free(neighbours_index), neighbours_index = NULL;

#if DYNAMIC_WEIGHT
            w = 0.9 - iter * ((0.9 - 0.4) / iteration);
            my_c = 0.1 - iter * (0.1 / (iteration / 2.0));

            if (my_c < 0)
                my_c = 0.0;

            s = 2.0 * drand48() * my_c; /** Seperation weight        **/
            a = 2.0 * drand48() * my_c; /** Alignment weight         **/
            c = 2.0 * drand48() * my_c; /** Cohesion weight          **/
            f = 2.0 * drand48();        /** Food attraction weight   **/
            e = my_c;                   /** Enemy distraction weight **/
#endif

            /** Update radius **/
            radius = (double)(upper_bound - lower_bound) / 8.0 + ((upper_bound - lower_bound) * ((double)iter / (double)iteration) * 4.0);

            /** Compute distance to food and pred **/
            distance(food_distance, dragonflies[df].position, food_pos, dim);
            distance(pred_distance, dragonflies[df].position, pred_pos, dim);

            /** Compute Separation **/
            if (neighbour_no > 0)
                separation_dragonfly(separation, dragonflies[df], neighbours, neighbour_no, dim);

            /** Compute Alignment **/
            if (neighbour_no > 0)
                alignment_dragonfly(alignment, neighbours, neighbour_no, dim);
            else
                memcpy(alignment, dragonflies[df].velocity, dim * sizeof(double));

            /** Compute Cohesion **/
            cohesion_dragonfly(cohesion, dragonflies[df], neighbours, neighbour_no, dim);

            /** Compute Food Attraction **/
            if (check_distance_radius(food_distance, radius, dim)) {
                food_attraction_dragonfly(food_attraction, dragonflies[df].position, food_pos, dim);
            }

            /** Compute Predator Distraction **/
            if (check_distance_radius(pred_distance, radius, dim))
                predator_distraction_dragonfly(pred_distraction, dragonflies[df].position, pred_pos, dim);

            /** Update Position and Velocity of Dragonfly **/
            if (neighbour_no >= 1) {
                for (int i = 0; i < dim; i++) {
                    if (food_near_dragonfly(food_distance, radius, dim)) {
                        dragonflies[df].velocity[i] = w * dragonflies[df].velocity[i] + (a * alignment[i] + c * cohesion[i] + s * separation[i]);
                        // dragonflies[df].velocity[i] = w * dragonflies[df].velocity[i] + drand48() * alignment[i] + drand48() * cohesion[i] + drand48() * separation[i];
                    } else {
                        dragonflies[df].velocity[i] = w * dragonflies[df].velocity[i] + (a * alignment[i] + c * cohesion[i] + s * separation[i] + f * food_attraction[i] + e * pred_distraction[i]);
                    }
                    // Control max velocity
                    if (dragonflies[df].velocity[i] > velocity_max)
                        dragonflies[df].velocity[i] = velocity_max;
                    if (dragonflies[df].velocity[i] < -velocity_max)
                        dragonflies[df].velocity[i] = -velocity_max;

                    dragonflies[df].position[i] = dragonflies[df].position[i] + dragonflies[df].velocity[i];
                }
            } else {
                levy_func(levy, dim, seed);
                for (int i = 0; i < dim; i++) {
                    dragonflies[df].position[i] += levy[i] * dragonflies[df].position[i];
                    dragonflies[df].velocity[i] = 0;
                }
            }

            /******* Check Boundaries *******/
            for (int axis = 0; axis < dim; axis++) {
                if (dragonflies[df].position[axis] > upper_bound) {
                    dragonflies[df].position[axis] = upper_bound;
                }
                if (dragonflies[df].position[axis] < lower_bound) {
                    dragonflies[df].position[axis] = lower_bound;
                }
            }

            for (int j = 0; j < dim; j++) {
                dragonflies[df].position[j] = (dragonflies[df].position[j] * (!(flagUb[j] + flagLb[j]))) + upper_bound * flagUb[j] + lower_bound * flagLb[j];
            }

            neighbour_no = 0;
            memset(neighbours, 0, sizeof(Neighbour) * neighbour_no);
            for (int i = 0; i < neighbour_no; i++) {
                printf(" %f ", neighbours[df].position[i]);
            }
        }

        while (1)
            ;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time += MPI_Wtime();

    MPI_Reduce(&elapsed_time, &global_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    result.best_score = food_value.value;
    memcpy(result.best_position, food_pos, dim * sizeof(double));

    if (!id) {
        printf("Function:   %s\n", obj_function_name(test_func));
        printf("Time:       %.4f\n", global_time);
        printf("Best Score: %.8f\nPosition:  ", result.best_score);
        print_array(result.best_position, dim);
        printf("\n");
    }

    elapsed_time = 0.0;
    global_time = 0.0;

    /** Free Memory **/
    // MPI_Type_free(&MPI_POSITION_TYPE);

    free(dragonflies), dragonflies = NULL;

    if (neighbour_no)
        freeNeighbours(neighbours, neighbour_no);

    free(neighbours), neighbours = NULL;

    free(food_pos), food_pos = NULL;
    free(pred_pos), pred_pos = NULL;

    free(food_distance), food_distance = NULL;
    free(pred_distance), pred_distance = NULL;

    free(local_food_pos), local_food_pos = NULL;
    free(local_pred_pos), local_pred_pos = NULL;

    free(dist), dist = NULL;
    free(levy), levy = NULL;
    free(separation), separation = NULL;
    free(alignment), alignment = NULL;
    free(cohesion), cohesion = NULL;
    free(food_attraction), food_attraction = NULL;
    free(pred_distraction), pred_distraction = NULL;

    free(flagLb), flagLb = NULL;
    free(flagUb), flagUb = NULL;
}
