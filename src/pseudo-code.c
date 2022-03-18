/**
 * > ----------------------------------------------------------------------------
 * >  DA Algorithm Pseudo Code                                                  |
 * > ----------------------------------------------------------------------------
 * > for iter: 0 to iteration:
 * >    for df: 0 to dragonflies_proc_no:
 * >      fitness function evaluation
 * >      food_fitness <- min{fitness values}
 * >      pred_fitness <- max{fitness values}
 * >    end
 * >
 * >    AllReduction to find global best and id of rank that find it
 * >    Broadcast to comunucate the position of best value (No Stopping)
 * >
 * >    for df: 0 to dragonfly_proc_no:
 * >        dragonflies.neighbor_no <- 0
 * >
 * >        for j: 0 to dragonfly_proc_no:
 * >            if df != j:
 * >                evaluate distance between df-th dragonfly and j-th dragonfly
 * >                if radius <= distance and distance != 0:
 * >                    add neighbours to neighbours list of df-th dragonfly
 * >                    add velocity and position of founded neighbours
 * >                endif
 * >            endif
 * >        end
 * >        MPI_Wait() Wait Result of Brocast Communication
 * >        Compute the behavior function
 * >        Check boundaries and max velocity
 * >        Update weight w,s,a,c,f,e
 * >        if df-th dragonfly have food in radius:
 * >            if df-th dragonflu have at least one neighbour:
 * >                Update velocity and position
 * >            else
 * >                Update position with levy function
 * >                Update velocity <- 0
 * >            end
 * >        end
 * >        Check Boundaries
 * >         MPI_Barriers()
 * >    end
 * > end
 */