#include "dragonflylib.h"
#include <mpi.h>

#define STATIC_WEIGHT 1

/********************************************************************************
 * mpirun -n 2 ./dragonfly_PA.o <iter> <dragonfly> <dim> <test_func> [<repeat>] *
 ********************************************************************************/
typedef struct hotspot_t {
    double value;
    int rank;
} HotSpot;

// void best_hotspot(HotSpot *h1, HotSpot *h2, int *len, MPI_Datatype *type);

void DA_Parallel(int id, int p, int iteration, int dragonfly_no, int dim, int test_func, Result result);

int main(int argc, char *argv[]) {
    int id, p;
    Result result;

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

    int repeat = argc == 6 ? atoi(argv[5]) : 1;

    int iteration = atoi(argv[1]);
    int dragonfly_no = atoi(argv[2]);
    int dim = atoi(argv[3]);
    int test_func = atoi(argv[4]);

    result.best_position = (double *)mycalloc(dim, sizeof(double), id);

    for (int r = 0; r < repeat; r++)
        DA_Parallel(id, p, iteration, dragonfly_no, dim, test_func, result);

    MPI_Finalize();
    return 0;
}

void DA_Parallel(int id, int p, int iteration, int dragonfly_no, int dim, int test_func, Result result) {

    int seed = 72024;

    srand48(seed);

    double elapsed_time = 0.0;
    double elapsed_time1 = 0.0;
    double global_time = 0.0;

    /* Bounds */
    int upper_bound = 0;
    int lower_bound = 0;

    /* Test Function */
    double (*fitness_function)(double *x, int dimension);
    fitness_function = func_obj(test_func, &upper_bound, &lower_bound);

    /* Init radius and max velocity */
    double radius = (upper_bound - lower_bound) / 20.0;
    double velocity_max = (upper_bound - lower_bound) / 20.0;

    /* Dragonflies for processor */
    int dragonfly_proc_no = BLOCK_SIZE(id, p, dragonfly_no);
    Dragonfly *dragonflies = (Dragonfly *)(double *)mycalloc(dragonfly_proc_no, sizeof(Dragonfly), id);
    init_struct_dragonfly(dragonflies, dragonfly_proc_no, dim, upper_bound, lower_bound, id, seed);

    int neighbour_no = 0;
    Neighbour *neighbours = NULL;
    init_neighbours(neighbours, dim);

    /*********************** Global Variable ***********************/
    HotSpot food_value, pred_value;

    /** Food & Predator Value **/
    food_value.rank = -1;
    food_value.value = INFINITY;

    pred_value.rank = -1;
    pred_value.value = -INFINITY;

    /** Food & Predator Position **/
    double *food_pos = (double *)mycalloc(dim, sizeof(double), id);
    double *pred_pos = (double *)mycalloc(dim, sizeof(double), id);

    double *local_food_pos = (double *)mycalloc(dim, sizeof(double), id);
    double *local_pred_pos = (double *)mycalloc(dim, sizeof(double), id);

    /*********************** Local Variable ***********************/
    HotSpot local_food_value, local_pred_value;

    /** Food & Predator Value **/
    local_food_value.rank = id;
    local_food_value.value = INFINITY;

    local_pred_value.rank = id;
    local_pred_value.value = -INFINITY;

    /** Food & Predator Distance **/
    double *food_distance = (double *)mycalloc(dim, sizeof(double), id);
    double *pred_distance = (double *)mycalloc(dim, sizeof(double), id);

    /** Distance between dragonflies **/
    double *dist = (double *)mycalloc(dim, sizeof(double), id);

    /** Levy Vector **/
    double *levy = (double *)mycalloc(dim, sizeof(double), id);

    /** Behaviours Vector **/
    double *separation = (double *)mycalloc(dim, sizeof(double), id);
    double *alignment = (double *)mycalloc(dim, sizeof(double), id);
    double *cohesion = (double *)mycalloc(dim, sizeof(double), id);
    double *food_attraction = (double *)mycalloc(dim, sizeof(double), id);
    double *pred_distraction = (double *)mycalloc(dim, sizeof(double), id);

    /** Define Weight **/
#if !STATIC_WEIGHT
    double w = 0.0;
    double s = 0.0;
    double a = 0.0;
    double c = 0.0;
    double f = 0.0;
    double e = 0.0;
    double my_c = 0.0;
#endif

#if STATIC_WEIGHT
    double w = drand48() * (0.9 - 0.2) + 0.2;
    double s = 0.1;
    double a = 0.1;
    double c = 0.7;
    double f = 0.1;
    double e = 0.1;
#endif

    /** MPI_Datatype User-Defined **/
    MPI_Datatype MPI_POSITION_TYPE;
    MPI_Type_contiguous(dim, MPI_DOUBLE, &MPI_POSITION_TYPE);
    MPI_Type_commit(&MPI_POSITION_TYPE);

    /** Non-Blocking AllReduce: Request & Status **/
    MPI_Request req;
    MPI_Status st;

    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time = -MPI_Wtime();

    /** Start DA Algorithm **/
    for (int iter = 0; iter <= iteration; iter++) {

        // MPI_Barrier(MPI_COMM_WORLD);
        // elapsed_time1 = -MPI_Wtime();

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
        MPI_Bcast(food_pos, 1, MPI_POSITION_TYPE, food_value.rank, MPI_COMM_WORLD);
        MPI_Bcast(pred_pos, 1, MPI_POSITION_TYPE, pred_value.rank, MPI_COMM_WORLD);

        /** Add neighbours **/
        for (int df = 0; df < dragonfly_proc_no; df++) {

            int index = 0;
            neighbour_no = 0;

            for (int j = 0; j < dragonfly_proc_no; j++) {

                dist = distance(dragonflies[df].position, dragonflies[j].position, dim);

                /** Validate and Add Neighbours to List of Neighbours **/
                if (validate_neighbour(dist, radius, dim)) {

                    neighbour_no += 1;
                    add_neighbours(&neighbours, neighbour_no, dim);
                    memcpy(neighbours[index].position, dragonflies[j].position, dim * sizeof(double));
                    memcpy(neighbours[index].velocity, dragonflies[j].velocity, dim * sizeof(double));
                    index += 1;
                }
            }
#if !STATIC_WEIGHT
            w = 0.9 - iter * ((0.9 - 0.4) / iteration);
            my_c = 0.1 - iter * (0.1 / (iteration / 2.0));

            if (my_c < 0)
                my_c = 0.0;

            s = 2 * drand48() * my_c; /** Seperation weight        **/
            a = 2 * drand48() * my_c; /** Alignment weight         **/
            c = 2 * drand48() * my_c; /** Cohesion weight          **/
            f = 2 * drand48();        /** Food attraction weight   **/
            e = my_c;                 /** Enemy distraction weight **/
#endif
            /** Update radius **/
            radius = (upper_bound - lower_bound) / 4.0 + ((upper_bound - lower_bound) * ((double)iter / iteration) * 2.0);

            /** Compute distance to food and pred **/
            food_distance = distance(dragonflies[df].position, food_pos, dim);
            pred_distance = distance(dragonflies[df].position, pred_pos, dim);

            /** Compute Separation **/
            if (neighbour_no > 0)
                separation_dragonfly(separation, dragonflies[df], neighbours, neighbour_no, dim);

            /** Compute Alignment **/
            if (neighbour_no > 0)
                alignment_dragonfly(alignment, neighbours, dim, neighbour_no);
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
                        dragonflies[df].velocity[i] = w * dragonflies[df].velocity[i] + (a * alignment[i] + c * cohesion[i] + s * separation[i] + f * food_attraction[i] + e * pred_distraction[i]);
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
                levy = levy_func(dim, seed);
                for (int i = 0; i < dim; i++) {
                    dragonflies[df].position[i] += levy[i] * dragonflies[df].position[i];
                    dragonflies[df].velocity[i] = 0;
                }
            }

            /******* Check Boundaries *******/
            for (int axis = 0; axis < dim; axis++) {
                if (dragonflies[df].position[axis] > upper_bound) {
                    dragonflies[df].position[axis] = lower_bound;
                }
                if (dragonflies[df].position[axis] < lower_bound) {
                    dragonflies[df].position[axis] = upper_bound;
                }
            }

            double *flagUb = (double *)calloc(dim, sizeof(double));
            double *flagLb = (double *)calloc(dim, sizeof(double));
            for (int j = 0; j < dim; j++) {
                dragonflies[df].position[j] = (dragonflies[df].position[j] * (!(flagUb[j] + flagLb[j]))) + upper_bound * flagUb[j] + lower_bound * flagLb[j];
            }
        }

        // MPI_Barrier(MPI_COMM_WORLD);
        // elapsed_time1 += MPI_Wtime();

        // printf("%d) Time: %.7f\n", id, elapsed_time1);
        // printf("%d) Food: %.7f\n", iter, food_value.value);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time += MPI_Wtime();

    MPI_Reduce(&elapsed_time, &global_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    result.best_score = food_value.value;
    memcpy(result.best_position, food_pos, dim * sizeof(double));

    if (!id) {
        printf("Time: %.4f\n", global_time);
        printf("Best Score: %.8f\nPosition: ", result.best_score);
        print_array(result.best_position, dim);
        printf("\n");
    }

    elapsed_time = 0.0;
    global_time = 0.0;

    /** Free Memory **/
    MPI_Type_free(&MPI_POSITION_TYPE);
    memset(&result, 0, sizeof(result));
    free(dragonflies), dragonflies = NULL;
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
}
