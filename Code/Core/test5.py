import numpy as np
from numba import njit

@njit
def _walk(
    point_x,
    point_y,
    t_index,
    vertices_ID,
    neighbour_ID,
    points,
    walk_pts,
    walk_nbr,
    wh_vtx
):
    '''
    Walks from the given triangle to the triangle enclosing the
    given point.

        point_id : The index (corresponding to the points array) of the
                   point to be inserted into the triangulation.
         t_index : The index of the triangle to start the walk from.
     vertices_ID : The global array storing all the indices (corresponding
                   to the points array) of the vertices of all the triangles.
    neighbour_ID : The global array storing the indices of the neighbouring
                   triangles.
          points : The global array storing the co-ordinates of all the
                   points to be triangulated.
    '''

    ## print("-- _walk called --")

    while True:
        # print("t_index : " + str(t_index))
        # print("neighbors of t_index : " + str(neighbour_ID[3*t_index:3*t_index+3]//3))
        for i in range(3):
            walk_pts[2*i] = points[2*vertices_ID[3*t_index+i]]
            walk_pts[2*i+1] = points[2*vertices_ID[3*t_index+i]+1]
            walk_nbr[i] = neighbour_ID[3*t_index+i]
            wh_vtx[i] = vertices_ID[3*t_index+i]

        wh_idx = 3
        for i in range(3):
            if wh_vtx[i] == -1:
                wh_idx = i
                break
        ## print("idx : " + str(idx))
        if wh_idx == 3:
            # i.e. t_index is a real triangle
            wh_t_op_index_in_t = 4
            a_x = walk_pts[0]
            a_y = walk_pts[1]
            b_x = walk_pts[2]
            b_y = walk_pts[3]
            c_x = walk_pts[4]
            c_y = walk_pts[5]
            temp1 = b_x*point_y
            temp2 = a_x*point_y
            temp3 = b_y*point_x
            temp4 = a_y*point_x
            temp5 = c_x*point_y
            temp6 = c_y*point_x

            # print("         : a = " + str([a_x, a_y]))
            # print("         : b = " + str([b_x, b_y]))
            # print("         : c = " + str([c_x, c_y]))

            if temp1 + temp4 + b_y*a_x < temp3 + temp2 + b_x*a_y:
                # signed area of a, b, p < 0
                # print("first term: " + str(temp1 + temp4 + b_y*a_x))
                # print("second term: " + str(temp3 + temp2 + b_x*a_y))
                # print("difference: " + str(temp1 + temp4 + b_y*a_x-(temp3 + temp2 + b_x*a_y)))
                wh_t_op_index_in_t = 2
            else:
                if temp5 + temp3 + c_y*b_x < temp6 + temp1 + c_x*b_y:
                    # print("first term: " + str(temp5 + temp3 + c_y*b_x))
                    # print("second term: " + str(temp6 + temp1 + c_x*b_y))
                    # print("difference: " + str(temp5 + temp3 + c_y*b_x-(temp6 + temp1 + c_x*b_y)))
                    # signed area of b, c, p < 0
                    wh_t_op_index_in_t = 0
                else:
                    if temp2 + temp6 + a_y*c_x < temp4 + a_x*c_y + temp5:
                        # print("first term: " + str(temp2 + temp6 + a_y*c_x))
                        # print("second term: " + str(temp4 + a_x*c_y + temp5))
                        # print("difference: " + str(temp2 + temp6 + a_y*c_x-(temp4 + a_x*c_y + temp5)))
                        # signed area of c, a, p < 0
                        wh_t_op_index_in_t = 1

            ## print("t_op_index_in_t : " + str(t_op_index_in_t))

            if wh_t_op_index_in_t != 4:
                wh_t_op_index = walk_nbr[wh_t_op_index_in_t]//3
            else:
                wh_t_op_index = t_index
        else:
            # i.e. t_index is a ghost triangle. In this case, simply step into the
            # adjacent non-ghost triangle
            wh_t_op_index = walk_nbr[wh_idx]//3

        # print("t_op_index : " + str(wh_t_op_index))
        ## print("neighbors of t_op_index : " + str(neighbour_ID[3*t_op_index:3*t_op_index+3]//3))
        if wh_t_op_index == t_index:
            break
        else:
            t_index = wh_t_op_index
            wh_idx = 3
                                                            #       Equivalent to:
            for i in range(3):                              #           np.where(
                if vertices_ID[3*wh_t_op_index+i] == -1:    #----->         vertices_ID[
                    wh_idx = i                              #----->             3*t_op_index:3*t_op_index+3
                    break                                   #               ] == -1    
                                                            #           )       
            if wh_idx != 3:
                # i.e. t_op_index is a ghost triangle
                break                

    ## print("-- _walk exited --")
    return t_index

# 0.6459151875, 0.12141810518238563]

@njit
def _cavity_helper(
    point_x,
    point_y,
    vtx,
    tri_points,
):
    '''
    Checks whether the given point lies inside the circumcircle
    of the given triangle. Returns True if it does.

        point : The co-ordinates of the point to be inserted into the
                triangulation.
          vtx : The indices corresponding to the global points array of the 
                vertices of t_index.
    tri_x_pts : The x co-ordinates of the vertices of the triangle.
    tri_y_pts : The y co-ordinates of the vertices of the triangle.
           sd : The sub-determinants of the triangle.
    '''


    idx = 3
    for i in range(3):
        if vtx[i] == -1:
            idx = i
            break
    ## print("ch_vtx : " + str(vtx))
    if idx != 3:
        ## print("ghost triangle in _cavity_helper")
        # i.e. t_index is a ghost triangle
        if idx == 0: 
            v1 = 1
            v2 = 2
        else:
            if idx == 1:
                v1 = 0
                v2 = 2
            else:
            # idx == 2:
                v1 = 0
                v2 = 1

        a_x = tri_points[2*v1]
        a_y = tri_points[2*v1+1]
        b_x = tri_points[2*v2]
        b_y = tri_points[2*v2+1]
        temp1 = (b_x-a_x)*(point_y-a_y)
        temp2 = (point_x-a_x)*(b_y-a_y)

        if temp1 == temp2:
            min_x = min(a_x, b_x)
            max_x = max(a_x, b_x)
            min_y = min(a_y, b_y)
            max_y = max(a_y, b_y)
            if min_x <= point_x and point_x <= max_x and min_y <= point_y and point_y <= max_y:
                return True
            else:
                return False
        elif idx == 1:
            # i.e. the 'real' points are stored non-contiguously
            if temp1 < temp2:
                return True
            else:
                return False
        else:
            if temp1 > temp2:
                return True
            else:
                return False
    else:
        ## print("real triangle in _cavity_helper")
        # i.e. t_index is a real triangle
        a_x = tri_points[0]
        a_y = tri_points[1]
        b_x = tri_points[2]
        b_y = tri_points[3]
        c_x = tri_points[4]
        c_y = tri_points[5]

        ax_ = a_x - point_x
        ay_ = a_y - point_y
        bx_ = b_x - point_x
        by_ = b_y - point_y
        cx_ = c_x - point_x
        cy_ = c_y - point_y

        temp = (ax_*ax_+ay_*ay_)*(bx_*cy_-cx_*by_) - (bx_*bx_+by_*by_)*(ax_*cy_-cx_*ay_) + (cx_*cx_+cy_*cy_)*(ax_*by_-bx_*ay_)

        if temp > 0:
            return True
        else:
            return False

@njit
def _identify_cavity(
    points,
    point_x,
    point_y,
    t_index,
    neighbour_ID,
    vertices_ID,
    ch_vtx,
    ch_pts,
    ic_bad_tri,
    ic_boundary_tri,
    ic_boundary_vtx,
):
    '''
    Identifies all the 'bad' triangles, i.e. the triangles whose circumcircles
    enclose the given point. Returns a list of the indices of the bad triangles
    and a list of the triangles bordering the cavity.
    
          points : The global array containing the co-ordinates of all the points to be
                   triangulated.
        point_id : The index corresponding to the points array of the point to be
                   inserted into the triangulation.
         t_index : The index of the triangle enclosing point_id.
    neighbour_ID : The global array containing the indices of the neighbours of
                   all the triangles.
     vertices_ID : The global array containing the indices (corresponding to the 
                   points array) of the vertices of all the triangles.
        sub_dets : The global array containing the sub-determinants of all the
                   traingles.
    '''

    ## print("-- _identify_cavity called --")
    ## print("point_id: " + str(point_id))
    ## print("t_index: " + str(t_index))

    ## print("neighbor_id : " + str(self.neighbour_ID[3*t_index:3*t_index + 3]//3))

    ## print("neighbors of enclosing_tri : " + str(neighbour_ID[3*t_index:3*t_index+3]//3))

    ic_len_bad_tri = len(ic_bad_tri)
    ic_bad_tri_end = 0

    ic_len_boundary_tri = len(ic_boundary_tri)
    ic_boundary_tri_end = 0

    ic_len_boundary_vtx = len(ic_boundary_vtx)
    ic_boundary_vtx_end = 0

    # Adding the first bad triangle, i.e. the enclosing triangle
    ic_bad_tri[ic_bad_tri_end] = t_index
    ic_bad_tri_end += 1

    ## print("t_index : " + str(t_index))
    ## print("neighbours of t_index : " + str(neighbour_ID[3*t_index:3*t_index+3]//3))
    for j in range(3):
        # j = index of neighbour, i.e. j'th neighbour
        # is given by neighbour_ID[3*t_index+j]//3
        for i in range(3):
            # Building the input arrays for _cavity_helper()
            ch_pts[2*i] = points[2*vertices_ID[3*(neighbour_ID[t_index*3+j]//3) + i]]
            ch_pts[2*i+1] = points[2*vertices_ID[3*(neighbour_ID[t_index*3+j]//3) + i]+1]
            ch_vtx[i] = vertices_ID[3*(neighbour_ID[t_index*3+j]//3) + i]

        temp_bool = _cavity_helper(
            point_x,
            point_y,
            ch_vtx,
            ch_pts,
        )
        if temp_bool == True:
            # i.e. the j'th neighbour is a bad triangle
            # i.e. j'th neighbour is not already stored in the bad_triangles list
            if ic_bad_tri_end >= ic_len_bad_tri:
                # checking if the array has space to accommodate another element
                temp_arr = np.empty(2*ic_len_bad_tri, dtype=np.int64)
                for l in range(ic_bad_tri_end):
                    temp_arr[l] = ic_bad_tri[l]
                ic_len_bad_tri = 2*ic_len_bad_tri
                ic_bad_tri = temp_arr

            ic_bad_tri[ic_bad_tri_end] = neighbour_ID[t_index*3+j]//3
            ic_bad_tri_end += 1
        else:
            # i.e. the j'th neighbour is a boundary triangle
            if ic_boundary_tri_end >= ic_len_boundary_tri:
                # checking if the array has space to accommodate another element
                temp_arr = np.empty(2*ic_len_boundary_tri, dtype=np.int64)
                for l in range(ic_boundary_tri_end):
                    temp_arr[l] = ic_boundary_tri[l]
                ic_len_boundary_tri = 2*ic_len_boundary_tri
                ic_boundary_tri = temp_arr

            ic_boundary_tri[ic_boundary_tri_end] = neighbour_ID[t_index*3+j]
            ic_boundary_tri_end += 1

            # Storing the vertices of t_index that lie on the boundary
            if ic_boundary_vtx_end >= ic_len_boundary_vtx:
                # checking if the array has space to accommodate another element
                temp_arr = np.empty(2*ic_len_boundary_vtx, dtype=np.int64)
                for l in range(ic_boundary_vtx_end):
                    temp_arr[l] = ic_boundary_vtx[l]
                ic_len_boundary_vtx = 2*ic_len_boundary_vtx
                ic_boundary_vtx = temp_arr

            ic_boundary_vtx[ic_boundary_vtx_end] = vertices_ID[3*t_index+(j+1)%3]
            ic_boundary_vtx[ic_boundary_vtx_end+1] = vertices_ID[3*t_index+(j+2)%3]
            ic_boundary_vtx_end += 2
    ## print("ic_bool_indices : " + str(ic_bool_indices))

    ic_iter = 1
    while ic_bad_tri_end > 1:
        ## print("while loop entered")

        t_index = ic_bad_tri[ic_iter]

        ## print("t_index : " + str(t_index))
        ## print("neighbours of t_index : " + str(neighbour_ID[3*t_index:3*t_index+3]//3))
        
        for j in range(3):

            tri_idx = neighbour_ID[t_index*3+j]//3

            for i in range(3):
                # Building the input arrays for _cavity_helper()
                ch_pts[2*i] = points[2*vertices_ID[3*tri_idx + i]]
                ch_pts[2*i+1] = points[2*vertices_ID[3*tri_idx + i]+1]
                ch_vtx[i] = vertices_ID[3*tri_idx + i]

            temp_bool = _cavity_helper(
                point_x,
                point_y,
                ch_vtx,
                ch_pts,
            )

            if temp_bool == True:
                # i.e. the j'th neighbour is a bad triangle

                already_stored = False
                for k in range(ic_bad_tri_end):
                    # Checking if it has already been stored in the array ic_bad_tri.
                    if ic_bad_tri[k] == tri_idx:
                        already_stored = True
                        break

                if already_stored == False:
                    # i.e. j'th neighbour is not already stored in the bad_triangles list
                    if ic_bad_tri_end >= ic_len_bad_tri:
                        # checking if the array has space to accommodate another element
                        temp_arr = np.empty(2*ic_len_bad_tri, dtype=np.int64)
                        for l in range(ic_bad_tri_end):
                            temp_arr[l] = ic_bad_tri[l]
                        ic_len_bad_tri = 2*ic_len_bad_tri
                        ic_bad_tri = temp_arr

                    ic_bad_tri[ic_bad_tri_end] = neighbour_ID[t_index*3+j]//3
                    ic_bad_tri_end += 1
            else:
                # i.e. the j'th neighbour is a boundary triangle

                if ic_boundary_tri_end >= ic_len_boundary_tri:
                    # checking if the array has space to accommodate another element
                    temp_arr = np.empty(2*ic_len_boundary_tri, dtype=np.int64)
                    for l in range(ic_boundary_tri_end):
                        temp_arr[l] = ic_boundary_tri[l]
                    ic_len_boundary_tri = 2*ic_len_boundary_tri
                    ic_boundary_tri = temp_arr

                ic_boundary_tri[ic_boundary_tri_end] = neighbour_ID[t_index*3+j]
                ic_boundary_tri_end += 1

                # Storing the vertices of t_index that lie on the boundary
                if ic_boundary_vtx_end >= ic_len_boundary_vtx:
                    # checking if the array has space to accommodate another element
                    temp_arr = np.empty(2*ic_len_boundary_vtx, dtype=np.int64)
                    for l in range(ic_boundary_vtx_end):
                        temp_arr[l] = ic_boundary_vtx[l]
                    ic_len_boundary_vtx = 2*ic_len_boundary_vtx
                    ic_boundary_vtx = temp_arr

                ic_boundary_vtx[ic_boundary_vtx_end] = vertices_ID[3*t_index+(j+1)%3]
                ic_boundary_vtx[ic_boundary_vtx_end+1] = vertices_ID[3*t_index+(j+2)%3]
                ic_boundary_vtx_end += 2

        ic_iter += 1

        ## print("ic_bool_indices : " + str(ic_bool_indices))
        ## print("idx : " + str(idx))
        ## print("length of bad_triangles : " + str(len(bad_triangles)))
        ## print("bad_triangles : " + str(bad_triangles))

        if ic_iter == ic_bad_tri_end:

            ## print("ic_boundary_vtx_end : " + str(ic_boundary_vtx_end))
            ## print("ic_boundary_tri_end : " + str(ic_boundary_tri_end))
            ## print("ic_bad_tri_end : " + str(ic_bad_tri_end))
            break

    ## print("ic_boundary_tri : " + str(ic_boundary_tri[0:ic_boundary_tri_end]//3))
    ## print("bad_triangles : " + str(bad_triangles))
    ## print("-- _identify_cavity exited --")

    return ic_bad_tri, ic_bad_tri_end, ic_boundary_tri, ic_boundary_tri_end, ic_boundary_vtx

@njit
def _make_Delaunay_ball(
    point_id,
    bad_triangles,
    bad_triangles_end,
    boundary_triangles,
    boundary_triangles_end,
    boundary_vtx,
    points,
    new_tri_indices,
    new_tri_vtx,
    new_tri_nbr,
):
    '''
    Joins all the vertices on the boundary to the new point, and forms
    the corresponding triangles along with their adjacencies. Returns the index
    of a new triangle, to be used as the starting point of the next walk.

              point_id : The index corresponding to the points array of the point
                         to be inserted into the triangulation.
         bad_triangles : The list fo traingles whose circumcircle contains point_id.
    boundary_triangles : The list of triangles lying on the boundary of the cavity
                         formed by the bad triangles.
          boundary_vtx : The vertices lying on the boundary of the cavity formed by
                         all the bad triangles.
                points : The global array storing the co-ordinates of all the points
                         to be triangulated.
             csd_final : The array to be passed to _calculate_sub_dets, it contains
                         the final sub-determinants. [shape: 3 x 1]
            csd_points : The array to be passed to _calculate_sub_dets, it contains
                         the co-ordinates of the points. [shape: 6 x 1]
    '''

    ## print("-- _make_Delaunay_ball called --")
    ## print("boundary_vertices : " + str(boundary_vtx))
    ## print("boundary_triangles : " + str(boundary_triangles//3))
    ## print("bad_triangles : " + str(bad_triangles[0:bad_triangles_end]))

    ## print("len(new_tri_indices) : " + str(len(new_tri_indices)))
    ## print("boundary_triangles_end : " + str(boundary_triangles_end))

    if len(new_tri_indices) < boundary_triangles_end:
        new_tri_indices = np.empty(boundary_triangles_end, dtype=np.int64)
        new_tri_vtx = np.empty(3*boundary_triangles_end, dtype=np.int64)
        new_tri_nbr = np.empty(3*boundary_triangles_end, dtype=np.int64)

    new_tri_indices_end = 0
    new_tri_vtx_end = 0
    new_tri_nbr_end = 0

    # populating the cavity with new triangles
    for i in range(boundary_triangles_end):
        new_tri_vtx[3*i] = point_id
        new_tri_vtx[3*i+1] = boundary_vtx[2*i]
        new_tri_vtx[3*i+2] = boundary_vtx[2*i+1]
        new_tri_vtx_end += 3

        new_tri_nbr[3*i] = boundary_triangles[i]
        new_tri_nbr[3*i+1] = -10 # placeholder value
        new_tri_nbr[3*i+2] = -10 # placeholder value
        new_tri_nbr_end += 3

        if i+1 <= bad_triangles_end:
            new_tri_indices[i] = bad_triangles[i]
            new_tri_indices_end += 1
            index_of_last_existing_tri = i
        else:
            new_tri_indices[i] = -10 # placeholder value
            new_tri_indices_end += 1

    ## print("new_tri_indices : " + str(new_tri_indices[0:new_tri_indices_end]))
    ## print("-- _make_Delaunay_ball exited --")

    return new_tri_indices, new_tri_indices_end, new_tri_nbr, new_tri_vtx, index_of_last_existing_tri

@njit
def assembly(
    old_tri,
    wh_vtx,
    walk_pts,
    walk_nbr,
    ic_bad_tri,
    ic_boundary_tri,
    ic_boundary_vtx,
    ch_pts,
    ch_vtx,
    mDb_new_tri_indices,
    mDb_new_tri_vtx,
    mDb_new_tri_nbr,
    points,
    vertices_ID,
    neighbour_ID,
    num_tri_v,
    num_tri_n
):
    for point_id in np.arange(3, int(len(points)/2)):
        ## print('---------- Loop no. '+ str(point_id-2) +' Starts ---------- ' )
        ## print("index of point being inserted: " + str(point_id))

        ## print("--- _walk called ---")
        enclosing_tri = _walk(
            points[2*point_id],   # point_x
            points[2*point_id+1], # point_y
            old_tri,              # t_index
            vertices_ID,          # vertices_ID
            neighbour_ID,         # neighbour_ID
            points,               # points
            walk_pts,             # walk_pts
            walk_nbr,             # walk_nbr
            wh_vtx                # wh_vtx
        )
        ## print("--- _walk exited ---")
        ## print("index of enclosing triangle: " + str(enclosing_tri))

        ic_bad_tri, ic_bad_tri_end, ic_boundary_tri, ic_boundary_tri_end, ic_boundary_vtx = _identify_cavity(
            points,               # points
            points[2*point_id],   # point_x
            points[2*point_id+1], # point_y
            enclosing_tri,        # t_index
            neighbour_ID,         # neighbour_ID
            vertices_ID,          # vertices_ID
            ch_vtx,               # ch_vtx
            ch_pts,               # ch_pts
            ic_bad_tri,           # ic_bad_tri
            ic_boundary_tri,      # ic_boundary_tri
            ic_boundary_vtx,      # ic_boundary_vtx
        )

        mDb_new_tri_indices, mDb_new_tri_indices_end, mDb_new_tri_nbr, mDb_new_tri_vtx, index_of_last_existing_tri = _make_Delaunay_ball(
            point_id,
            ic_bad_tri,                # ic_bad_tri
            ic_bad_tri_end,            # ic_bad_tri_end
            ic_boundary_tri,           # ic_boundary_tri
            ic_boundary_tri_end,       # ic_boundary_tri_end
            ic_boundary_vtx,           # ic_boundary_vtx
            points,
            mDb_new_tri_indices,
            mDb_new_tri_vtx,
            mDb_new_tri_nbr,
        )
        ## print("new_T_nbr : " + str(new_T_nbr))
        ## print(new_T_nbr.shape)
        ## print("new_T_idx (before changes): " + str(new_T_idx))
        ## print("idx_last_existing_tri: " + str(idx_last_existing_tri))
        for i in np.arange(index_of_last_existing_tri+1):
            ## print("-- i = " + str(i) + " --")
            t_index = mDb_new_tri_indices[i]
            neighbour_ID[3*t_index:3*t_index+3] = mDb_new_tri_nbr[3*i:3*i+3]
            vertices_ID[3*t_index:3*t_index+3] = mDb_new_tri_vtx[3*i:3*i+3]
            neighbour_ID[ic_boundary_tri[i]] = 3*t_index
        for i in np.arange(index_of_last_existing_tri+1, mDb_new_tri_indices_end):
            ## print("-- i = " + str(i) + " --")
            neighbour_ID[3*num_tri_n:3*num_tri_n+3] = mDb_new_tri_nbr[3*i:3*i+3]
            vertices_ID[3*num_tri_v:3*num_tri_v+3] = mDb_new_tri_vtx[3*i:3*i+3]
            neighbour_ID[ic_boundary_tri[i]] = 3*num_tri_n
            num_tri_v += 1
            num_tri_n += 1
            mDb_new_tri_indices[i] = num_tri_n-1

        ## print("new_T_idx (after changes): " + str(new_T_idx))
        temp_len = len(mDb_new_tri_indices)

        for i in range(temp_len):
            if i == mDb_new_tri_indices_end:
                break
            else:
                t1 = mDb_new_tri_indices[i]
                for j in range(temp_len):
                    if j == mDb_new_tri_indices_end:
                        break
                    else:
                        t2 = mDb_new_tri_indices[j]
                        if vertices_ID[3*t1+1] == vertices_ID[3*t2+2]:
                            ## print("vtx of " + str(t1) + " : " + str(self.vertices_ID[3*t1:3*t1+3]))
                            ## print("vtx of " + str(t2) + " : " + str(self.vertices_ID[3*t2:3*t2+3]))
                            neighbour_ID[3*t1+2] = 3*t2+1
                            neighbour_ID[3*t2+1] = 3*t1+2
                            ## print("adjacency bw " + str(t1) + " and " + str(t2) + " created.")
                            break

        old_tri = enclosing_tri

    return vertices_ID, neighbour_ID, num_tri_v, num_tri_n

@njit
def add_point(
    old_tri,
    wh_vtx,
    walk_pts,
    walk_nbr,
    ic_bad_tri,
    ic_boundary_tri,
    ic_boundary_vtx,
    ch_pts,
    ch_vtx,
    mDb_new_tri_indices,
    mDb_new_tri_vtx,
    mDb_new_tri_nbr,
    points,
    point_id,
    num_points,
    vertices_ID,
    neighbour_ID,
    num_tri_v,
    num_tri_n
):

    point_already_exists = False
    for i in range(num_points):
        if i != point_id and points[2*i] == points[2*point_id] and points[2*i+1] == points[2*point_id+1]:
            point_already_exists = True
            break

    if point_already_exists == False:
        enclosing_tri = _walk(
            points[2*point_id],   # point_x
            points[2*point_id+1], # point_y
            old_tri,              # t_index
            vertices_ID,          # vertices_ID
            neighbour_ID,         # neighbour_ID
            points,               # points
            walk_pts,             # walk_pts
            walk_nbr,             # walk_nbr
            wh_vtx                # wh_vtx
        )
        # print("--- _walk exited ---")
        ## print("index of enclosing triangle: " + str(enclosing_tri))

        ic_bad_tri, ic_bad_tri_end, ic_boundary_tri, ic_boundary_tri_end, ic_boundary_vtx = _identify_cavity(
            points,               # points
            points[2*point_id],   # point_x
            points[2*point_id+1], # point_y
            enclosing_tri,        # t_index
            neighbour_ID,         # neighbour_ID
            vertices_ID,          # vertices_ID
            ch_vtx,               # ch_vtx
            ch_pts,               # ch_pts
            ic_bad_tri,           # ic_bad_tri
            ic_boundary_tri,      # ic_boundary_tri
            ic_boundary_vtx,      # ic_boundary_vtx
        )

        mDb_new_tri_indices, mDb_new_tri_indices_end, mDb_new_tri_nbr, mDb_new_tri_vtx, index_of_last_existing_tri = _make_Delaunay_ball(
            point_id,
            ic_bad_tri,                # ic_bad_tri
            ic_bad_tri_end,            # ic_bad_tri_end
            ic_boundary_tri,           # ic_boundary_tri
            ic_boundary_tri_end,       # ic_boundary_tri_end
            ic_boundary_vtx,           # ic_boundary_vtx
            points,
            mDb_new_tri_indices,
            mDb_new_tri_vtx,
            mDb_new_tri_nbr,
        )
        ## print("new_T_nbr : " + str(new_T_nbr))
        ## print(new_T_nbr.shape)
        ## print("new_T_idx (before changes): " + str(new_T_idx))
        ## print("idx_last_existing_tri: " + str(idx_last_existing_tri))
        for i in np.arange(index_of_last_existing_tri+1):
            ## print("-- i = " + str(i) + " --")
            t_index = mDb_new_tri_indices[i]
            neighbour_ID[3*t_index:3*t_index+3] = mDb_new_tri_nbr[3*i:3*i+3]
            vertices_ID[3*t_index:3*t_index+3] = mDb_new_tri_vtx[3*i:3*i+3]
            neighbour_ID[ic_boundary_tri[i]] = 3*t_index
        for i in np.arange(index_of_last_existing_tri+1, mDb_new_tri_indices_end):
            ## print("-- i = " + str(i) + " --")
            if 3*num_tri_n >= len(neighbour_ID):
                #checking if the array has space for another element
                temp_arr_1 = np.empty(2*len(neighbour_ID), dtype=np.int64)
                temp_arr_2= np.empty(2*len(vertices_ID), dtype=np.int64)
                for l in range(len(neighbour_ID)):
                    temp_arr_1[l] = neighbour_ID[l]
                    temp_arr_2[l] = vertices_ID[l]
                neighbour_ID = temp_arr_1
                vertices_ID = temp_arr_2
            neighbour_ID[3*num_tri_n:3*num_tri_n+3] = mDb_new_tri_nbr[3*i:3*i+3]
            vertices_ID[3*num_tri_v:3*num_tri_v+3] = mDb_new_tri_vtx[3*i:3*i+3]
            neighbour_ID[ic_boundary_tri[i]] = 3*num_tri_n
            num_tri_v += 1
            num_tri_n += 1
            mDb_new_tri_indices[i] = num_tri_n-1

        ## print("new_T_idx (after changes): " + str(new_T_idx))
        temp_len = len(mDb_new_tri_indices)

        for i in range(temp_len):
            if i == mDb_new_tri_indices_end:
                break
            else:
                t1 = mDb_new_tri_indices[i]
                for j in range(temp_len):
                    if j == mDb_new_tri_indices_end:
                        break
                    else:
                        t2 = mDb_new_tri_indices[j]
                        if vertices_ID[3*t1+1] == vertices_ID[3*t2+2]:
                            ## print("vtx of " + str(t1) + " : " + str(self.vertices_ID[3*t1:3*t1+3]))
                            ## print("vtx of " + str(t2) + " : " + str(self.vertices_ID[3*t2:3*t2+3]))
                            neighbour_ID[3*t1+2] = 3*t2+1
                            neighbour_ID[3*t2+1] = 3*t1+2
                            ## print("adjacency bw " + str(t1) + " and " + str(t2) + " created.")
                            break

    return vertices_ID, neighbour_ID, num_tri_v, num_tri_n, point_already_exists

@njit
def initialize(
    points,
    vertices_ID,
    neighbour_ID
):
    N = int(len(points)/2)

    a_x = points[0]
    a_y = points[1]
    b_x = points[2]
    b_y = points[3]

    num_tri_v = np.int64(0)
    num_tri_n = np.int64(0)
    num_tri_s = np.int64(0)

    idx = 2
    while True:
        p_x = points[2*idx]
        p_y = points[2*idx+1]
        signed_area = 0.5*(
            (b_x-a_x)*(p_y-a_y)-
            (p_x-a_x)*(b_y-a_y)
        )
        if signed_area > 0:
            ## print("signed_area : " + str(signed_area))
            points[4], points[2*idx] = points[2*idx], points[4]
            points[4+1], points[2*idx+1] = points[2*idx+1], points[4+1]
            break
        elif signed_area < 0:
            ## print("signed_area : " + str(signed_area))
            points[4], points[2*idx] = points[2*idx], points[4]
            points[4+1], points[2*idx+1] = points[2*idx+1], points[4+1]
            points[0], points[2] = points[2], points[0]
            points[1], points[3] = points[3], points[1]
            break
        else:
            idx += 1

    for i in range(3):
        vertices_ID[i] = i
    num_tri_v += 1

    vertices_ID[3] = 0      #---|
    vertices_ID[4] = -1     #   |---> 1st triangle [ghost]
    vertices_ID[5] = 1      #---|

    vertices_ID[6] = 1      #---|
    vertices_ID[7] = -1     #   |---> 2nd triangle [ghost]
    vertices_ID[8] = 2      #---|

    vertices_ID[9] = 2      #---|
    vertices_ID[10] = -1    #   |---> 3rd triangle [ghost]
    vertices_ID[11] = 0     #---|

    num_tri_v += 3

    neighbour_ID[0] = 3*2+1     #---|
    neighbour_ID[1] = 3*3+1     #   |---> 0th triangle [real]
    neighbour_ID[2] = 3*1+1     #---|

    neighbour_ID[3] = 3*2+2     #---|
    neighbour_ID[4] = 3*0+2     #   |---> 1st triangle [ghost]
    neighbour_ID[5] = 3*3+0     #---|

    neighbour_ID[6] = 3*3+2     #---|
    neighbour_ID[7] = 3*0+0     #   |---> 2nd triangle [ghost]
    neighbour_ID[8] = 3*1+0     #---|

    neighbour_ID[9] = 3*1+2     #---|
    neighbour_ID[10] = 3*0+1    #   |---> 3rd triangle [ghost]
    neighbour_ID[11] = 3*2+0    #---|

    num_tri_n += 4

    return points, vertices_ID, neighbour_ID, num_tri_v, num_tri_n


class Delaunay2D:

    def __init__(self, points):
        # points: list of points to be triangulated

        N = int(len(points)/2)

        self.vertices_ID = np.empty(3*(2*N-2), dtype=np.int64)
        self.neighbour_ID = np.empty(3*(2*N-2), dtype=np.int64)

        self.points, self.vertices_ID, self.neighbour_ID, self.num_tri_v, self.num_tri_n = initialize(
            points,
            self.vertices_ID,
            self.neighbour_ID,
        )

    def makeDT(self):
        # makes the Delaunay traingulation of the given point list
        # self.plotDT()

        old_tri = np.int64(0)

        # Arrays that will be passed into the jit-ed functions
        # so that they don't have to get their hands dirty with
        # object creation.
        ### _walk_helper
        self.wh_vtx = np.empty(3, dtype=np.int64)
        ### _walk
        self.walk_pts = np.empty(6, dtype=np.float64)
        self.walk_nbr = np.empty(3, dtype=np.int64)
        ### _identify_cavity
        self.ic_bad_tri = np.empty(50, dtype=np.int64)
        self.ic_boundary_tri = np.empty(50, dtype=np.int64)
        self.ic_boundary_vtx = np.empty(2*50, dtype=np.int64)
        ### _cavity_helper
        self.ch_pts = np.empty(6, dtype=np.float64)
        self.ch_vtx = np.empty(3, dtype=np.int64)
        ### _make_Delaunay_ball
        self.mDb_new_tri_indices = np.empty(50, dtype=np.int64)
        self.mDb_new_tri_vtx = np.empty(3*50, dtype=np.int64)
        self.mDb_new_tri_nbr = np.empty(3*50, dtype=np.int64)
        ### adjacency creation for newly created triangles
        #adjacencies = np.empty(2*50, dtype=np.int64)
        
        self.vertices_ID, self.neighbour_ID, self.num_tri_v, self.num_tri_n = assembly(
            old_tri,
            self.wh_vtx,
            self.walk_pts,
            self.walk_nbr,
            self.ic_bad_tri,
            self.ic_boundary_tri,
            self.ic_boundary_vtx,
            self.ch_pts,
            self.ch_vtx,
            self.mDb_new_tri_indices,
            self.mDb_new_tri_vtx,
            self.mDb_new_tri_nbr,
            self.points,
            self.vertices_ID,
            self.neighbour_ID,
            self.num_tri_v,
            self.num_tri_n
        )

        return

    def add_point(self, point_x, point_y):

        self.points = np.append(self.points, [point_x, point_y])
        old_tri = int(-1)
        self.vertices_ID, self.neighbour_ID, self.num_tri_v, self.num_tri_n = add_point(
            old_tri,
            self.wh_vtx,
            self.walk_pts,
            self.walk_nbr,
            self.ic_bad_tri,
            self.ic_boundary_tri,
            self.ic_boundary_vtx,
            self.ch_pts,
            self.ch_vtx,
            self.mDb_new_tri_indices,
            self.mDb_new_tri_vtx,
            self.mDb_new_tri_nbr,
            self.points,
            self.vertices_ID,
            self.neighbour_ID,
            self.num_tri_v,
            self.num_tri_n
        )

        return


    def exportDT(self):
        # Export the present Delaunay triangulation

        points = self.points
        vertices = np.array([], dtype=np.int64)
        for i in np.arange(0, self.num_tri_v):
            idx = np.where(self.vertices_ID[3*i:3*i+3]==-1)[0]
            if len(idx) == 0:
                # i.e. i is not a ghost triangle
                vertices = np.append(vertices, self.vertices_ID[3*i:3*i+3])
        return points, vertices

    def _centroid(self, a, b, c):
        return (a[0]+b[0]+c[0])/3, (a[1]+b[1]+c[1])/3

    def plotDT(self):
        # Plots the present Delaunay triangulation

        import matplotlib.pyplot as plt
        points, vertices = self.exportDT()
        ## print("vertices : " + str(vertices))
        ## print("points : " + str(points))

        plt.triplot(
            points[0::2],
            points[1::2],
            vertices.reshape(int(len(vertices)/3), 3)
        )

        plt.axis('equal')
        return plt.gca

def perf(N):
    points = np.random.rand(2*N)
    DT = Delaunay2D(points)
    DT.makeDT()

if __name__ == "__main__":
    import sys
    perf(int(sys.argv[1]))