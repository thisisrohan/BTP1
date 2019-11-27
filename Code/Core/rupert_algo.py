import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from .test5 import Delaunay2D, add_point, _walk


@njit
def find_and_fix_bad_segments(
    segments,
    num_segments,
    segments_to_delete,
    new_segments,
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
    num_points,
    vertices_ID,
    num_tri_v,
    neighbour_ID,
    num_tri_n,
    plt_iter,
    org_num_points
):

    # if len(segments_to_analyse) >= len(segments):
    #     segments_to_analyse[0:2*num_segments] = segments[0:2*num_segments].copy()
    # else:
    #     segments_to_analyse = segments.copy()
    # print("--------- find_and_fix_bad_segments ---------")
    seg_iter = 0
    no_more_segs_to_split = True
    num_segments_to_delete = 0
    while seg_iter < num_segments:
        a_idx = segments[2*seg_iter]
        b_idx = segments[2*seg_iter+1]
        a_x = points[2*a_idx]
        a_y = points[2*a_idx+1]
        b_x = points[2*b_idx]
        b_y = points[2*b_idx+1]
        center_x = 0.5*(a_x + b_x)
        center_y = 0.5*(a_y + b_y)
        radius_square = 0.25*((a_x-b_x)*(a_x-b_x)+(a_y-b_y)*(a_y-b_y))
        is_encroached = False

        for i in range(num_points):
            if i == a_idx or i == b_idx:
                continue
            else:
                p_x = points[2*i]
                p_y = points[2*i+1]
                temp = (center_x-p_x)*(center_x-p_x)+(center_y-p_y)*(center_y-p_y)
                if temp < radius_square:
                    is_encroached = True
                    break

        if is_encroached == True:
            # print("segment midpoint to be inserted")
            # print("segment : " + str([a_x, a_y]) + " and " + str([b_x, b_y]))
            # print("center : " + str([center_x, center_y]))
            # print("encraoching points : " + str([p_x, p_y]))
            no_more_segs_to_split = False

            if 2*num_points >= len(points):
                # checking if the array has space to accommodate another element
                temp_arr_points = np.empty(2*len(points), dtype=np.float64)
                for l in range(2*num_points):
                    temp_arr_points[l] = points[l]
                points = temp_arr_points

            if a_idx < org_num_points and b_idx < org_num_points:
                pass
            elif a_idx < org_num_points and b_idx >= org_num_points:
                # a is an input vertex
                shell_number = int(np.round(np.log2(radius_square**0.5/0.01)))
                angle = np.arctan2(b_y-a_y, b_x-a_x)
                radius = radius_square**0.5
                # center_x = a_x + 0.01*(2**shell_number)*np.cos(angle)
                # center_y = a_y + 0.01*(2**shell_number)*np.sin(angle)
                center_x = a_x*(2*radius - 0.01*(2**shell_number))/(2*radius) + b_x*(0.01*(2**shell_number))/(2*radius)
                center_y = a_y*(2*radius - 0.01*(2**shell_number))/(2*radius) + b_y*(0.01*(2**shell_number))/(2*radius)
            elif a_idx >= org_num_points and b_idx < org_num_points:
                # b is an input vertex
                shell_number = int(np.round(np.log2(radius_square**0.5/0.01)))
                angle = np.arctan2(a_y-b_y, a_x-b_x)
                radius = radius_square**0.5
                # center_x = b_x + 0.01*(2**shell_number)*np.cos(angle)
                # center_y = b_y + 0.01*(2**shell_number)*np.sin(angle)
                center_x = b_x*(2*radius - 0.01*(2**shell_number))/(2*radius) + a_x*(0.01*(2**shell_number))/(2*radius)
                center_y = b_y*(2*radius - 0.01*(2**shell_number))/(2*radius) + a_y*(0.01*(2**shell_number))/(2*radius)

            points[2*num_points] = center_x
            points[2*num_points+1] = center_y
            num_points += 1

            old_tri = num_tri_n-1
            point_id = num_points-1
            vertices_ID, neighbour_ID, num_tri_v, num_tri_n, point_already_exists = add_point(
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
            )
            # print("segment midpoint inserted")

            # plt.clf()
            # final_vertices = np.empty_like(vertices_ID)
            # final_vertices = remove_ghost_tri(
            #     vertices_ID,
            #     num_tri_v,
            #     final_vertices
            # )
            # num_tri = int(len(final_vertices)/3)
            # # for t_index in range(num_tri):
            # #     r_idx = final_vertices[3*t_index]
            # #     s_idx = final_vertices[3*t_index+1]
            # #     t_idx = final_vertices[3*t_index+2]
            # #     r_x = points[2*r_idx]
            # #     r_y = points[2*r_idx+1]
            # #     s_x = points[2*s_idx]
            # #     s_y = points[2*s_idx+1]
            # #     t_x = points[2*t_idx]
            # #     t_y = points[2*t_idx+1]
            # #     if (r_x == s_x and s_x == t_x) or (r_y == s_y and s_y == t_y):
            # #         print("t_index : " + str(t_index))
            # #         print("vertices : " + str([r_idx, s_idx, t_idx]))
            # #         print("         : a = " + str([r_x, r_y]))
            # #         print("         : b = " + str([s_x, s_y]))
            # #         print("         : c = " + str([t_x, t_y]))
            # plt.triplot(
            #    points[0::2],
            #    points[1::2],
            #    final_vertices.reshape(num_tri, 3)
            # )
            # plt.plot([a_x, b_x], [a_y, b_y], color='k')
            # # # for m in np.arange(0, num_tri, dtype=np.int64):
            # # #     a = [points[2*final_vertices[3*m]], points[2*final_vertices[3*m]+1]]
            # # #     b = [points[2*final_vertices[3*m+1]], points[2*final_vertices[3*m+1]+1]]
            # # #     c = [points[2*final_vertices[3*m+2]], points[2*final_vertices[3*m+2]+1]]
            # # #     plt.plot([a[0], b[0]], [a[1], b[1]], color='k', linewidth=0.5)
            # # #     plt.plot([b[0], c[0]], [b[1], c[1]], color='k', linewidth=0.5)
            # # #     plt.plot([c[0], a[0]], [c[1], a[1]], color='k', linewidth=0.5)
            # plt.plot(center_x, center_y, 'o', color='red', markersize=10)
            # if point_already_exists == True:
            #     plt.title("[find_and_fix_bad_segments] segment midpoint (not added, already exists)")
            # else:
            #     plt.title("[find_and_fix_bad_segments] segment midpoint")
            # # plt.text(center_x, center_y, "this point")
            # plt.axis('equal')
            # plt.savefig(str(plt_iter)+".png", dpi=300, bbox_inches='tight')
            # plt_iter += 1
            # # plt.show()

            if 2*num_segments+2 >= len(segments):
                # checking if the array has space to accommodate another element
                temp_arr_segs = np.empty(2*len(segments), dtype=np.int64)
                for l in range(len(segments)):
                    temp_arr_segs[l] = segments[l]
                segments = temp_arr_segs

            segments[2*num_segments] = a_idx
            segments[2*num_segments+1] = num_points-1
            num_segments += 1
            segments[2*num_segments] = num_points-1
            segments[2*num_segments+1] = b_idx
            num_segments += 1

            if num_segments_to_delete >= len(segments_to_delete):
                temp_seg_to_delete = np.empty(2*num_segments_to_delete, dtype=np.int64)
                for l in range(num_segments_to_delete):
                    temp_seg_to_delete[l] = segments_to_delete[l]
                segments_to_delete = temp_seg_to_delete

            segments_to_delete[num_segments_to_delete] = seg_iter
            num_segments_to_delete += 1

            # if 2*num_segments_to_analyse >= len(segments_to_analyse):
            #     # checking if the array has space to accommodate another element
            #     temp_arr_segs2analyse = np.empty(2*len(segments_to_analyse), dtype=np.int64)
            #     for l in range(2*num_segments_to_analyse):
            #         temp_arr_segs2analyse[l] = segments_to_analyse[l]
            #     segments_to_analyse = temp_arr_segs2analyse

            # segments_to_analyse[2*num_segments_to_analyse] = a_idx
            # segments_to_analyse[2*num_segments_to_analyse+1] = num_points-1
            # num_segments_to_analyse += 1
            # segments_to_analyse[2*num_segments_to_analyse] = num_points-1
            # segments_to_analyse[2*num_segments_to_analyse+1] = b_idx
            # num_segments_to_analyse += 1

        seg_iter += 1

    if no_more_segs_to_split == False:
        # print("segments_to_delete : " + str(segments_to_delete))
        # print("num_segments_to_delete : " + str(num_segments_to_delete))
        # print("segments : ")
        # print(str(segments[0:2*num_segments].reshape(num_segments, 2)))
        segments_to_delete_end = 0
        new_seg_iter = 0
        for i in range(num_segments):
            if i != segments_to_delete[segments_to_delete_end]:
                if 2*new_seg_iter >= len(new_segments):
                    temp_new_seg = np.empty(2*len(new_segments), dtype=np.int64)
                    for l in range(len(new_segments)):
                        temp_new_seg[l] = new_segments[l]
                    new_segments = temp_new_seg
                new_segments[2*new_seg_iter] = segments[2*i]
                new_segments[2*new_seg_iter+1] = segments[2*i+1]
                new_seg_iter += 1
            else:
                segments_to_delete_end += 1
                if segments_to_delete_end == num_segments_to_delete:
                    segments_to_delete_end -= 1

        return new_segments, new_seg_iter, points, num_points, neighbour_ID, num_tri_n, vertices_ID, num_tri_v, no_more_segs_to_split, plt_iter
    else:
        return segments, num_segments, points, num_points, neighbour_ID, num_tri_n, vertices_ID, num_tri_v, no_more_segs_to_split, plt_iter


@njit
def find_and_fix_bad_triangles(
    segments,
    num_segments,
    segments_to_delete,
    new_segments,
    encroached_segments,
    min_angle,
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
    num_points,
    vertices_ID,
    num_tri_v,
    neighbour_ID,
    num_tri_n,
    plt_iter,
    org_num_points
):


    # print("--------- find_and_fix_bad_triangles ---------")
    tri_iter = 0
    no_more_tri_to_split = True
    while tri_iter < num_tri_v:
        is_ghost = False
        if vertices_ID[3*tri_iter] == -1:
            is_ghost = True
        elif vertices_ID[3*tri_iter+1] == -1:
            is_ghost = True
        elif vertices_ID[3*tri_iter+2] == -1:
            is_ghost = True

        if is_ghost == False:
            a_idx = vertices_ID[3*tri_iter]
            b_idx = vertices_ID[3*tri_iter+1]
            c_idx = vertices_ID[3*tri_iter+2]
            a_x = points[2*a_idx]
            a_y = points[2*a_idx+1]
            b_x = points[2*b_idx]
            b_y = points[2*b_idx+1]
            c_x = points[2*c_idx]
            c_y = points[2*c_idx+1]
            a_sq = (b_x-c_x)*(b_x-c_x)+(b_y-c_y)*(b_y-c_y)
            b_sq = (c_x-a_x)*(c_x-a_x)+(c_y-a_y)*(c_y-a_y)
            c_sq = (a_x-b_x)*(a_x-b_x)+(a_y-b_y)*(a_y-b_y)
            ## print("a : " + str(a) + ", b : " + str(b) + ", c : " + str(c))
            temp_max = max(a_sq, b_sq, c_sq)
            if b_sq == temp_max:
                a_idx, b_idx = b_idx, a_idx
                a_x, b_x = b_x, a_x
                a_y, b_y = b_y, a_y
                a_sq, b_sq = b_sq, a_sq
            elif c_sq == temp_max:
                a_idx, c_idx = c_idx, a_idx
                a_x, c_x = c_x, a_x
                a_y, c_y = c_y, a_y
                a_sq, c_sq = c_sq, a_sq
            temp_max = max(b_sq, c_sq)
            if c_sq == temp_max:
                c_idx, b_idx = b_idx, c_idx
                c_x, b_x = b_x, c_x
                c_y, b_y = b_y, c_y
                c_sq, b_sq = b_sq, c_sq
            a = np.sqrt(a_sq)
            b = np.sqrt(b_sq)
            c = np.sqrt(c_sq)
            if c <= 0.1*a:
                temp_mu = c-(a-b)
                C = 2*np.arctan(((((a-b)+c)*temp_mu)/((a+(b+c))*((a-c)+b)))**0.5)
                A = np.arccos(((c_sq+(b_sq-a_sq))/(2*c))/a)
                B = (np.pi - A) - C
            else:
                C = np.arccos(((b/a)+(a/b)-(c/a)*(c/b))*0.5)
                A = np.arccos(((c_sq+(b_sq-a_sq))/(2*c))/a)
                B = (np.pi - A) - C
            if A < min_angle or B < min_angle or C < min_angle:
                no_more_tri_to_split = False
                # tri_iter is a skinny triangle

                temp = np.sin(2*A) + np.sin(2*B) + np.sin(2*C)
                if temp != 0:
                    circumcenter_x = (a_x*np.sin(2*A)+b_x*np.sin(2*B)+c_x*np.sin(2*C))/temp
                    circumcenter_y = (a_y*np.sin(2*A)+b_y*np.sin(2*B)+c_y*np.sin(2*C))/temp
                else:
                    # print("temp == 0")
                    if b_y == a_y:
                        # m_1 = inf
                        b_x, c_x = c_x, b_x
                        b_y, c_y = c_y, b_y
                    if b_y == c_y:
                        # m_2 = inf
                        b_x, a_x = a_x, b_x
                        b_y, a_y = a_y, b_y
                    m_1 = (a_x-b_x)/(b_y-a_y)
                    x_1 = 0.5*(a_x+b_x)
                    y_1 = 0.5*(a_y+b_y)
                    m_2 = (b_x-c_x)/(c_y-b_y)
                    x_2 = 0.5*(b_x+c_x)
                    y_2 = 0.5*(b_y+c_y)
                    circumcenter_x = (y_2-y_1)/(m_1-m_2)+(m_1*x_1-m_2*x_2)/(m_1-m_2)
                    circumcenter_y = m_1*(y_2-m_2*x_2)/(m_1-m_2) + m_2*(m_1*x_1-y_1)/(m_1-m_2)

                encroached_segments_end = 0

                for i in range(num_segments):
                    # segment between k'th point and h'th point
                    h_idx = segments[2*i]
                    k_idx = segments[2*i+1]
                    h_x = points[2*h_idx]
                    h_y = points[2*h_idx+1]
                    k_x = points[2*k_idx]
                    k_y = points[2*k_idx+1]
                    center_x = 0.5*(h_x + k_x)
                    center_y = 0.5*(h_y + k_y)
                    radius_square = 0.25*((h_x-k_x)*(h_x-k_x)+(h_y-k_y)*(h_y-k_y))
                    temp = (center_x-circumcenter_x)*(center_x-circumcenter_x)+(center_y-circumcenter_y)*(center_y-circumcenter_y)
                    if temp < radius_square:
                        # circumcenter encroaches upon segment i
                        if encroached_segments_end >= len(encroached_segments):
                            # checking if the array has space to accommodate another element
                            temp_arr_encroached_segs = np.empty(2*len(encroached_segments), dtype=np.int64)
                            for l in range(encroached_segments_end):
                                temp_arr_encroached_segs[l] = encroached_segments[l]
                            encroached_segments = temp_arr_encroached_segs

                        encroached_segments[encroached_segments_end] = i
                        encroached_segments_end += 1

                if encroached_segments_end > 0:
                    # num_segments_to_delete = 0
                    for i in range(encroached_segments_end):
                        # print("segment midpoint inserted")
                        seg_idx = encroached_segments[i]
                        h_idx = segments[2*seg_idx]
                        k_idx = segments[2*seg_idx+1]
                        h_x = points[2*h_idx]
                        h_y = points[2*h_idx+1]
                        k_x = points[2*k_idx]
                        k_y = points[2*k_idx+1]
                        center_x = 0.5*(h_x + k_x)
                        center_y = 0.5*(h_y + k_y)

                        if 2*num_points >= len(points):
                            # checking if the array has space to accommodate another element
                            temp_arr_points = np.empty(2*len(points), dtype=np.float64)
                            for l in range(2*num_points):
                                temp_arr_points[l] = points[l]
                            points = temp_arr_points

                        if h_idx < org_num_points and k_idx < org_num_points:
                            pass
                        elif h_idx < org_num_points and k_idx >= org_num_points:
                            # h is an input vertex
                            radius = (0.25*((h_x-k_x)*(h_x-k_x)+(h_y-k_y)*(h_y-k_y)))**0.5
                            shell_number = int(np.round(np.log2(radius/0.01)))
                            angle = np.arctan2(k_y-h_y, k_x-h_x)
                            # center_x = h_x + 0.01*(2**shell_number)*np.cos(angle)
                            # center_y = h_y + 0.01*(2**shell_number)*np.sin(angle)
                            center_x = h_x*(2*radius - 0.01*(2**shell_number))/(2*radius) + k_x*(0.01*(2**shell_number))/(2*radius)
                            center_y = h_y*(2*radius - 0.01*(2**shell_number))/(2*radius) + k_y*(0.01*(2**shell_number))/(2*radius)
                        elif h_idx >= org_num_points and k_idx < org_num_points:
                            # k is an input vertex
                            radius = (0.25*((h_x-k_x)*(h_x-k_x)+(h_y-k_y)*(h_y-k_y)))**0.5
                            shell_number = int(np.round(np.log2(radius/0.01)))
                            angle = np.arctan2(h_y-k_y, h_x-k_x)
                            # center_x = k_x + 0.01*(2**shell_number)*np.cos(angle)
                            # center_y = k_y + 0.01*(2**shell_number)*np.sin(angle)
                            center_x = k_x*(2*radius - 0.01*(2**shell_number))/(2*radius) + h_x*(0.01*(2**shell_number))/(2*radius)
                            center_y = k_y*(2*radius - 0.01*(2**shell_number))/(2*radius) + h_y*(0.01*(2**shell_number))/(2*radius)
                        
                        points[2*num_points] = center_x
                        points[2*num_points+1] = center_y
                        num_points += 1

                        old_tri = num_tri_n-1
                        point_id = num_points-1
                        vertices_ID, neighbour_ID, num_tri_v, num_tri_n, point_already_exists = add_point(
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
                        )


                        if 2*num_segments >= len(segments):
                            # checking if the array has space to accommodate another element
                            temp_arr_segs = np.empty(2*len(segments), dtype=np.int64)
                            for l in range(len(segments)):
                                temp_arr_segs[l] = segments[l]
                            segments = temp_arr_segs

                        # segments[2*seg_idx] = h_idx
                        segments[2*seg_idx+1] = point_id
                        # num_segments += 1
                        segments[2*num_segments] = point_id
                        segments[2*num_segments+1] = k_idx
                        num_segments += 1

                        # if num_segments_to_delete >= len(segments_to_delete):
                        #     temp_seg_to_delete = np.empty(2*num_segments_to_delete, dtype=np.int64)
                        #     for l in range(num_segments_to_delete):
                        #         temp_seg_to_delete[l] = segments_to_delete[l]
                        #     segments_to_delete = temp_seg_to_delete

                        # segments_to_delete[num_segments_to_delete] = seg_idx
                        # num_segments_to_delete += 1

                        # plt.clf()
                        # final_vertices = np.empty_like(vertices_ID)
                        # final_vertices = remove_ghost_tri(
                        #     vertices_ID,
                        #     num_tri_v,
                        #     final_vertices
                        # )
                        # num_tri = int(len(final_vertices)/3)
                        # plt.triplot(
                        #    points[0::2],
                        #    points[1::2],
                        #    final_vertices.reshape(num_tri, 3)
                        # )
                        # plt.plot([h_x, k_x], [h_y, k_y], color='k')
                        # # for m in np.arange(0, num_tri, dtype=np.int64):
                        # #     a = [points[2*final_vertices[3*m]], points[2*final_vertices[3*m]+1]]
                        # #     b = [points[2*final_vertices[3*m+1]], points[2*final_vertices[3*m+1]+1]]
                        # #     c = [points[2*final_vertices[3*m+2]], points[2*final_vertices[3*m+2]+1]]
                        # #     plt.plot([a[0], b[0]], [a[1], b[1]], color='k', linewidth=0.5)
                        # #     plt.plot([b[0], c[0]], [b[1], c[1]], color='k', linewidth=0.5)
                        # #     plt.plot([c[0], a[0]], [c[1], a[1]], color='k', linewidth=0.5)
                        # plt.plot(center_x, center_y, 'o', color='red', markersize=10)
                        # if point_already_exists == True:
                        #     plt.title("[find_and_fix_bad_triangles] segment midpoint (not added, already exists)")
                        # else:
                        #     plt.title("[find_and_fix_bad_triangles] segment midpoint")
                        # plt.axis('equal')
                        # # plt.text(center_x, center_y, "this point")
                        # plt.savefig(str(plt_iter)+".png", dpi=300, bbox_inches='tight')
                        # plt_iter += 1
                        # # plt.show()

                    # segments_to_delete_end = 0
                    # new_seg_iter = 0
                    # for i in range(num_segments):
                    #     if i != segments_to_delete[segments_to_delete_end]:
                    #         if 2*new_seg_iter >= len(new_segments):
                    #             temp_new_seg = np.empty(2*len(new_segments), dtype=np.int64)
                    #             for l in range(len(new_segments)):
                    #                 temp_new_seg[l] = new_segments[l]
                    #             new_segments = temp_new_seg
                    #         new_segments[2*new_seg_iter] = segments[2*i]
                    #         new_segments[2*new_seg_iter+1] = segments[2*i+1]
                    #         new_seg_iter += 1
                    #     else:
                    #         segments_to_delete_end += 1

                else:
                    # print("circumcenter inserted")
                    if 2*num_points >= len(points):
                        # checking if the array has space to accommodate another element
                        temp_arr_points = np.empty(2*len(points), dtype=np.float64)
                        for l in range(2*num_points):
                            temp_arr_points[l] = points[l]
                        points = temp_arr_points

                    points[2*num_points] = circumcenter_x
                    points[2*num_points+1] = circumcenter_y
                    num_points += 1

                    old_tri = num_tri_n-1
                    point_id = num_points-1

                    # print("circumcenter_x : " + str(circumcenter_x) + ", circumcenter_y : " + str(circumcenter_y))
                    # print("t_index : " + str(tri_iter))
                    # print("vertices : " + str([a_idx, b_idx, c_idx]))
                    # print("         : a = " + str([a_x, a_y]))
                    # print("         : b = " + str([b_x, b_y]))
                    # print("         : c = " + str([c_x, c_y]))

                    vertices_ID, neighbour_ID, num_tri_v, num_tri_n, point_already_exists = add_point(
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
                    )
                    # if point_already_exists == True:
                    #     temp = np.sin(2*A) + np.sin(2*B) + np.sin(2*C)
                    #     circumcenter_x = (a_x*np.sin(2*A)+b_x*np.sin(2*B)+c_x*np.sin(2*C))/temp
                    #     circumcenter_y = (a_y*np.sin(2*A)+b_y*np.sin(2*B)+c_y*np.sin(2*C))/temp
                    #     vertices_ID, neighbour_ID, num_tri_v, num_tri_n, point_already_exists = t5.add_point(
                    #         old_tri,
                    #         wh_vtx,
                    #         walk_pts,
                    #         walk_nbr,
                    #         ic_bad_tri,
                    #         ic_boundary_tri,
                    #         ic_boundary_vtx,
                    #         ch_pts,
                    #         ch_vtx,
                    #         mDb_new_tri_indices,
                    #         mDb_new_tri_vtx,
                    #         mDb_new_tri_nbr,
                    #         points,
                    #         point_id,
                    #         num_points,
                    #         vertices_ID,
                    #         neighbour_ID,
                    #         num_tri_v,
                    #         num_tri_n
                    #     )


                    # plt.fill([a_x, b_x, c_x], [a_y, b_y, c_y], alpha=0.5, facecolor='C1')
                    # plt.plot(circumcenter_x, circumcenter_y, 's', color='green', markersize=10)
                    # plt.savefig(str(plt_iter-1)+".png", dpi=300, bbox_inches='tight')
                    # plt.clf()
                    # final_vertices = np.empty_like(vertices_ID)
                    # final_vertices = remove_ghost_tri(
                    #     vertices_ID,
                    #     num_tri_v,
                    #     final_vertices
                    # )
                    # num_tri = int(len(final_vertices)/3)
                    # plt.triplot(
                    #    points[0::2],
                    #    points[1::2],
                    #    final_vertices.reshape(num_tri, 3)
                    # )
                    # # plt.plot([h_x, k_x], [h_y, k_y], color='k')
                    # # for m in np.arange(0, num_tri, dtype=np.int64):
                    # #     a = [points[2*final_vertices[3*m]], points[2*final_vertices[3*m]+1]]
                    # #     b = [points[2*final_vertices[3*m+1]], points[2*final_vertices[3*m+1]+1]]
                    # #     c = [points[2*final_vertices[3*m+2]], points[2*final_vertices[3*m+2]+1]]
                    # #     plt.plot([a[0], b[0]], [a[1], b[1]], color='k', linewidth=0.5)
                    # #     plt.plot([b[0], c[0]], [b[1], c[1]], color='k', linewidth=0.5)
                    # #     plt.plot([c[0], a[0]], [c[1], a[1]], color='k', linewidth=0.5)
                    # # plt.plot(circumcenter_x, circumcenter_y, 's', col or='green', markersize=10)
                    # if point_already_exists == True:
                    #     plt.title("[find_and_fix_bad_triangles] circumcenter (not added, already exists)")
                    # else:
                    #     plt.title("[find_and_fix_bad_triangles] circumcenter")
                    # # plt.text(circumcenter_x, circumcenter_y, "this point")
                    # plt.axis('equal')
                    # # plt.savefig(str(plt_iter)+".png", dpi=300, bbox_inches='tight')
                    # plt_iter += 1
                    # # plt.show()

        tri_iter += 1

    return segments, num_segments, points, num_points, neighbour_ID, num_tri_n, vertices_ID, num_tri_v, no_more_tri_to_split, plt_iter

@njit
def insert_virus(
    insertion_points,
    vertices_ID,
    num_tri_v,
    neighbour_ID,
    num_tri_n,
    points,
    num_points,
    segments,
    num_segments,
    walk_pts,
    walk_nbr,
    wh_vtx,
    tri_to_be_deleted,
    final_vertices,
):
    '''
    insertion_points : 2k x 1 array (virus introduced at these k points)
    '''


    num_viral_points = int(0.5*len(insertion_points))

    num_tri_to_be_deleted = 0
    old_tri = 0
    for k in range(num_viral_points):
        insertion_point_x = insertion_points[2*k]
        insertion_point_y = insertion_points[2*k+1]

        enclosing_tri = _walk(
            insertion_point_x,
            insertion_point_y,
            old_tri,
            vertices_ID,
            neighbour_ID,
            points,
            walk_pts,
            walk_nbr,
            wh_vtx
        )

        old_tri = enclosing_tri

        if num_tri_to_be_deleted >= len(tri_to_be_deleted):
            # checking if the array has space for another element
            temp_arr_del_tri = np.empty(2*num_tri_to_be_deleted, dtype=np.int64)
            for l in range(num_tri_to_be_deleted):
                temp_arr_del_tri[l] = tri_to_be_deleted[l]
            tri_to_be_deleted = temp_arr_del_tri

        tri_to_be_deleted[num_tri_to_be_deleted] = enclosing_tri
        num_tri_to_be_deleted += 1

        last_tri = enclosing_tri

        tri_iter = 0
        while tri_iter < num_tri_to_be_deleted:
            # print("tri_iter : " + str(tri_iter))
            # print("num_tri_to_be_deleted : " + str(num_tri_to_be_deleted))

            tri_idx = tri_to_be_deleted[tri_iter]

            a_idx = vertices_ID[3*tri_idx]
            b_idx = vertices_ID[3*tri_idx+1]
            c_idx = vertices_ID[3*tri_idx+2]

            nbr_a = neighbour_ID[3*tri_idx]//3
            del_nbr_a = True
            nbr_b = neighbour_ID[3*tri_idx+1]//3
            del_nbr_b = True
            nbr_c = neighbour_ID[3*tri_idx+2]//3
            del_nbr_c = True

            for i in range(num_segments):
                h_idx = segments[2*i]
                k_idx = segments[2*i+1]

                if (h_idx == a_idx and k_idx == b_idx) or (h_idx == b_idx and k_idx == a_idx):
                    del_nbr_c = False
                if (h_idx == b_idx and k_idx == c_idx) or (h_idx == c_idx and k_idx == b_idx):
                    del_nbr_a = False
                if (h_idx == c_idx and k_idx == a_idx) or (h_idx == a_idx and k_idx == c_idx):
                    del_nbr_b = False

            for i in range(num_tri_to_be_deleted):
                temp_tri = tri_to_be_deleted[i]
                if nbr_a == temp_tri:
                    del_nbr_a = False
                if nbr_b == temp_tri:
                    del_nbr_b = False
                if nbr_c == temp_tri:
                    del_nbr_c = False

            if del_nbr_a == True:
                if num_tri_to_be_deleted >= len(tri_to_be_deleted):
                    # checking if the array has space for another element
                    temp_arr_del_tri = np.empty(2*num_tri_to_be_deleted, dtype=np.int64)
                    for l in range(num_tri_to_be_deleted):
                        temp_arr_del_tri[l] = tri_to_be_deleted[l]
                    tri_to_be_deleted = temp_arr_del_tri

                tri_to_be_deleted[num_tri_to_be_deleted] = nbr_a
                num_tri_to_be_deleted += 1

            if del_nbr_b == True:
                if num_tri_to_be_deleted >= len(tri_to_be_deleted):
                    # checking if the array has space for another element
                    temp_arr_del_tri = np.empty(2*num_tri_to_be_deleted, dtype=np.int64)
                    for l in range(num_tri_to_be_deleted):
                        temp_arr_del_tri[l] = tri_to_be_deleted[l]
                    tri_to_be_deleted = temp_arr_del_tri

                tri_to_be_deleted[num_tri_to_be_deleted] = nbr_b
                num_tri_to_be_deleted += 1

            if del_nbr_c == True:
                if num_tri_to_be_deleted >= len(tri_to_be_deleted):
                    # checking if the array has space for another element
                    temp_arr_del_tri = np.empty(2*num_tri_to_be_deleted, dtype=np.int64)
                    for l in range(num_tri_to_be_deleted):
                        temp_arr_del_tri[l] = tri_to_be_deleted[l]
                    tri_to_be_deleted = temp_arr_del_tri

                tri_to_be_deleted[num_tri_to_be_deleted] = nbr_c
                num_tri_to_be_deleted += 1

            tri_iter += 1
            # if tri_iter > 2:
            #     break

    # print("tri_to_be_deleted : " + str(tri_to_be_deleted[0:num_tri_to_be_deleted]))
    # for i in range(num_tri_to_be_deleted):
    #     t_index = tri_to_be_deleted[i]
    #     print("t_index : " + str(t_index))
    #     print("    vertices : " + str(vertices_ID[3*t_index:3*t_index+3]))
    #     print("           a : " + str([points[2*vertices_ID[3*t_index]], points[2*vertices_ID[3*t_index]+1]]))
    #     print("           b : " + str([points[2*vertices_ID[3*t_index+1]], points[2*vertices_ID[3*t_index+1]+1]]))
    #     print("           c : " + str([points[2*vertices_ID[3*t_index+2]], points[2*vertices_ID[3*t_index+2]+1]]))

    num_tri_final = 0
    for i in range(num_tri_v):
        store_in_final_array = True

        for j in range(num_tri_to_be_deleted):
            if i == tri_to_be_deleted[j]:
                store_in_final_array = False
                break

        if store_in_final_array == True:
            final_vertices[3*num_tri_final] = vertices_ID[3*i]
            final_vertices[3*num_tri_final+1] = vertices_ID[3*i+1]
            final_vertices[3*num_tri_final+2] = vertices_ID[3*i+2]

            num_tri_final += 1

    return final_vertices[0:3*num_tri_final]

#@njit
def assembly(
    segments,
    num_segments,
    segments_to_delete,
    new_segments,
    encroached_segments,
    min_angle,
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
    num_points,
    vertices_ID,
    num_tri_v,
    neighbour_ID,
    num_tri_n,
):

    org_num_points = num_points
    plt_iter = 1
    while True:
        # print("segments before find_and_fix_bad_segments : ")
        # print(str(segments[0:2*num_segments].reshape(num_segments, 2)))
        segments, num_segments, points, num_points, neighbour_ID, num_tri_n, vertices_ID, num_tri_v, no_more_segs_to_split, plt_iter = find_and_fix_bad_segments(
            segments,
            num_segments,
            segments_to_delete,
            new_segments,
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
            num_points,
            vertices_ID,
            num_tri_v,
            neighbour_ID,
            num_tri_n,
            plt_iter,
            org_num_points
        )
        # print("--------- find_and_fix_bad_segments exited ---------")
        # print("segments after find_and_fix_bad_segments : ")
        # print(str(segments[0:2*num_segments].reshape(num_segments, 2)))
        segments, num_segments, points, num_points, neighbour_ID, num_tri_n, vertices_ID, num_tri_v, no_more_tri_to_split, plt_iter = find_and_fix_bad_triangles(
            segments,
            num_segments,
            segments_to_delete,
            new_segments,
            encroached_segments,
            min_angle,
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
            num_points,
            vertices_ID,
            num_tri_v,
            neighbour_ID,
            num_tri_n,
            plt_iter,
            org_num_points
        )
        # print("--------- find_and_fix_bad_triangles exited ---------")
        # print("segments after find_and_fix_bad_triangles : ")
        # print(str(segments[0:2*num_segments].reshape(num_segments, 2)))
        if no_more_tri_to_split == True and no_more_segs_to_split == True:
            break

    return points[0:2*num_points], neighbour_ID[0:3*num_tri_n], vertices_ID[0:3*num_tri_v], segments[0:2*num_segments]

@njit
def remove_ghost_tri(
    vertices_ID,
    num_tri_v,
    final_vertices
):

    num_tri_final = 0
    for i in np.arange(0, num_tri_v):
        is_ghost = False
        if vertices_ID[3*i] == -1:
            is_ghost = True
        elif vertices_ID[3*i+1] == -1:
            is_ghost = True
        elif vertices_ID[3*i+2] == -1:
            is_ghost = True
        if is_ghost == False:
            # i.e. i is not a ghost triangle
            final_vertices[3*num_tri_final] = vertices_ID[3*i]
            final_vertices[3*num_tri_final+1] = vertices_ID[3*i+1]
            final_vertices[3*num_tri_final+2] = vertices_ID[3*i+2]
            num_tri_final += 1

    return final_vertices[0:3*num_tri_final]


class RefinedDelaunay:

    def __init__(
        self,
        points,
        segments,
        insertion_points,
        min_angle=10
    ):
        '''
          points : 2N x 1
        segments : M x 2
        '''

        temp = np.random.rand(20)
        tempDT = Delaunay2D(temp)
        tempDT.makeDT()
        del tempDT
        del temp

        if len(insertion_points.shape) > 1:
            insertion_points = insertion_points.reshape(2*len(insertion_points))

        DT = Delaunay2D(points)
        DT.makeDT()
        self.points = DT.points
        self.neighbour_ID = DT.neighbour_ID
        self.vertices_ID = DT.vertices_ID

        self.num_points = int(0.5*len(self.points))
        self.num_tri_v = int(len(self.vertices_ID)/3)
        self.num_tri_n = int(len(self.neighbour_ID)/3)
        self.num_segments = len(segments)

        self.segments = np.empty(4*self.num_segments, dtype=np.int64)
        self.segments[0:2*self.num_segments] = segments.reshape(2*self.num_segments)

        for i in range(self.num_segments):
            a_idx = self.segments[2*i]
            b_idx = self.segments[2*i+1]
            a_x = self.points[2*a_idx]
            a_y = self.points[2*a_idx+1]
            b_x = self.points[2*b_idx]
            b_y = self.points[2*b_idx+1]
            plt.plot([a_x, b_x], [a_y, b_y], linewidth=2, color='k')

        for i in range(self.num_points):
            plt.plot(self.points[2*i], self.points[2*i+1], '.', color='brown')


        plt.axis('equal')
        plt.title("Initial Point Set and the Constraints")
        plt.savefig("initial.png", dpi=300, bbox_inches='tight')

        final_vertices = np.empty_like(self.vertices_ID)
        final_vertices = remove_ghost_tri(
            self.vertices_ID,
            self.num_tri_v,
            final_vertices
        )
        num_tri = int(len(final_vertices)/3)
        plt.clf()
        plt.triplot(
            self.points[0::2],
            self.points[1::2],
            final_vertices.reshape(num_tri, 3),
            linewidth=0.75
        )
        plt.axis('equal')
        plt.title("Delaunay triangulation of initial point set")
        plt.savefig("00.png", dpi=300, bbox_inches='tight')
        # plt.show()

        self.min_angle = np.pi*min_angle/180

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
        ### find_and_fix_bad_segments
        self.segments_to_delete = np.empty(4*self.num_segments, dtype=np.int64)
        self.new_segments = np.empty(4*self.num_segments, dtype=np.int64)
        ### find_and_fix_bad_triangles
        self.encroached_segments = np.empty(4*self.num_segments, dtype=np.int64)    


        self.points, self.neighbour_ID, self.vertices_ID, self.segments = assembly(
            self.segments,
            self.num_segments,
            self.segments_to_delete,
            self.new_segments,
            self.encroached_segments,
            self.min_angle,
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
            self.num_points,
            self.vertices_ID,
            self.num_tri_v,
            self.neighbour_ID,
            self.num_tri_n
        )

        self.num_points = int(0.5*len(self.points))
        self.num_tri_v = int(len(self.vertices_ID)/3)
        self.num_tri_n = int(len(self.neighbour_ID)/3)
        self.num_segments = int(0.5*len(self.segments))

        ### insert_virus
        final_vertices = self.vertices_ID.copy()
        tri_to_be_deleted = np.empty(self.num_tri_v, dtype=np.int64)

        self.vertices_ID = insert_virus(
            insertion_points,
            self.vertices_ID,
            self.num_tri_v,
            self.neighbour_ID,
            self.num_tri_n,
            self.points,
            self.num_points,
            self.segments,
            self.num_segments,
            self.walk_pts,
            self.walk_nbr,
            self.wh_vtx,
            tri_to_be_deleted,
            final_vertices,
        )

        self.num_tri_v = int(len(self.vertices_ID)/3)


    def plotDT(self):
        final_vertices = np.empty_like(self.vertices_ID)
        final_vertices = remove_ghost_tri(
            self.vertices_ID,
            self.num_tri_v,
            final_vertices
        )
        num_tri = int(len(final_vertices)/3)
        plt.clf()
        plt.triplot(
            self.points[0::2],
            self.points[1::2],
            final_vertices.reshape(num_tri, 3),
            linewidth=0.75
        )
        for i in range(self.num_segments):
            a_idx = self.segments[2*i]
            b_idx = self.segments[2*i+1]
            a_x = self.points[2*a_idx]
            a_y = self.points[2*a_idx+1]
            b_x = self.points[2*b_idx]
            b_y = self.points[2*b_idx+1]
            plt.plot([a_x, b_x], [a_y, b_y], linewidth=2, color='k')

        plt.axis('equal')
        plt.title("Final Triangulation")
        plt.savefig("final.png", dpi=300, bbox_inches='tight')
        # plt.show()



def make_data():
    # points = np.array([
    #     0.5, 0,
    #     4, 0,
    #     6, 2,
    #     0, 4,
    #     1.5, 0.5,
    #     3, 0.5,
    #     3, 1.5,
    #     1, 1.5,
    #     1, 2,
    #     1, 3, 
    #     2, 2
    # ])
    # segments = np.array([
    #     [4, 5],
    #     [5, 6],
    #     [6, 7],
    #     [7, 4],
    #     [0, 1],
    #     [1, 2],
    #     [2, 3],
    #     [3, 0],
    #     [8, 9],
    #     [9, 10],
    #     [10, 8]
    # ])
    # insertion_points = np.array([[2, 1], [1.1, 2.1]])
    # points = np.array([
    #     0, 0,
    #     1, 0,
    #     2, 0.5,
    #     3, 4,
    #     3, 5,
    #     2, 3,
    #     1, 4,
    #     2, 5,
    #     -1, 5,
    #     -1, 3, 
    #     1, 2,
    #     1, 1,
    #     # -5, -5,
    #     # 20, -5,
    #     # 20, 10,
    #     # -5, 10
    # ])
    # segments = np.array([
    #     [0, 1],
    #     [1, 2],
    #     [2, 3],
    #     [3, 4],
    #     [4, 5],
    #     [5, 6],
    #     [6, 7],
    #     [7, 8],
    #     [8, 9],
    #     [9, 10],
    #     [10, 11],
    #     [11, 0],
    #     # [12, 13],
    #     # [13, 14],
    #     # [14, 15],
    #     # [15, 12]
    # ])
    # insertion_points = np.array([0, 1])
    # points = np.array([
    #     0, 0,
    #     4, 0,
    #     4, 0.25,
    #     4, 0.5,
    #     4, 0.75,
    #     4, 1,
    #     4, 1.25,
    #     4, 1.5,
    #     4, 1.75,
    #     4, 2,
    #     4, 2.25,
    #     4, 2.5,
    #     4, 2.75,
    #     4, 3,
    #     0, 3
    # ])
    # segments = np.array([
    #     [0, 1],
    #     [1, 2],
    #     [2, 3],
    #     [3, 4],
    #     [4, 5],
    #     [5, 6],
    #     [6, 7],
    #     [7, 8],
    #     [8, 9],
    #     [9, 10],
    #     [10, 11],
    #     [11, 12],
    #     [12, 13],
    #     [13, 14],
    #     [14, 0]
    # ])
    # insertion_points = np.array([-1, 1])
    # points = 2*np.array([
    #     [0, 0],
    #     [0.051673, -0.014696552140281],
    #     [0.091043, -0.010076888237317],
    #     [0.120571, -0.004775805350042],
    #     [0.15502, 0.002348107051437],
    #     [0.199311, 0.011722536638535],
    #     [0.238681, 0.019467808525461],
    #     [0.280512, 0.026813946153663],
    #     [0.327264, 0.033801721445369],
    #     [0.356791, 0.037465229088082],
    #     [0.396161, 0.041377339370863],
    #     [0.43061, 0.043921863378774],
    #     [0.46998, 0.045929673463347],
    #     [0.504429, 0.047025465882686],
    #     [0.54872, 0.04763133198028],
    #     [0.583169, 0.047510713731955],
    #     [0.617618, 0.046871583225751],
    #     [0.656988, 0.045478849114091],
    #     [0.691437, 0.043636195070778],
    #     [0.725886, 0.04119194505892],
    #     [0.765256, 0.037668154787437],
    #     [0.799705, 0.033980982002866],
    #     [1, 0],
    #     [0.765256, 0.089657930233398],
    #     [0.725886, 0.100782773538959],
    #     [0.691437, 0.110032733067277],
    #     [0.656988, 0.118794121075188],
    #     [0.617618, 0.128123842345224],
    #     [0.583169, 0.135601270805556],
    #     [0.54872, 0.142340760456029],
    #     [0.504429, 0.149644630770221],
    #     [0.46998, 0.153953703014286],
    #     [0.43061, 0.157182393169817],
    #     [0.396161, 0.158457311722946],
    #     [0.356791, 0.158092729977226],
    #     [0.327264, 0.156509949910914],
    #     [0.280512, 0.151610048134506],
    #     [0.238681, 0.144616286473781],
    #     [0.199311, 0.135662517154915],
    #     [0.15502, 0.122458931825376],
    #     [0.120571, 0.109361591025835],
    #     [0.091043, 0.095598458819934],
    #     [0.051673, 0.071816746190796],
    #     [0.031988, 0.055318035275167],
    #     [-1, -1],
    #     [2, -1],
    #     [2, 1],
    #     [-1, 1]
    # ])-[1, 0]
    
    # segments = [[i, i+1] for i in range(len(points)-1-4)]
    # segments.append([len(points)-1-4, 0])
    # segments.append([len(points)-1-3, len(points)-1-2])
    # segments.append([len(points)-1-2, len(points)-1-1])
    # segments.append([len(points)-1-1, len(points)-1-0])
    # segments.append([len(points)-1-0, len(points)-1-3])
    # points = points.reshape(2*len(points))
    # points = np.round(points, 4)
    # segments = np.array(segments)
    # insertion_points = 2*np.array([0.051673, 0])-[1, 0]
    points_1 = np.array([
        0.0, 0.0,
        .0075, .0176,
        .0125, .0215,
        .0250, .0276,
        .0375, .0316,
        .0500, .0347,
        .0750, .0394,
        .1000, .0428,
        .1250, .0455,
        .1500, .0476,
        .1750, .0493,
        .2000, .0507,
        .2500, .0528,
        .3000, .0540,
        .3500, .0547,
        .4000, .0550,
        .4500, .0548,
        .5000, .0543,
        .5500, .0533,
        .5750, .0527,
        .6000, .0519,
        .6250, .0511,
        .6500, .0501,
        .6750, .0489,
        .7000, .0476,
        .7250, .0460,
        .7500, .0442,
        .7750, .0422,
        .8000, .0398,
        .8250, .0370,
        .8500, .0337,
        .8750, .0300,
        .9000, .0255,
        .9250, .0204,
        .9500, .0144,
        .9750, .0074,
        1.0000, -.0008,
    ])
    points_2 = np.array([
        .0075, -.0176,
        .0125, -.0216,
        .0250, -.0281,
        .0375, -.0324,
        .0500, -.0358,
        .0750, -.0408,
        .1000, -.0444,
        .1250, -.0472,
        .1500, -.0493,
        .1750, -.0510,
        .2000, -.0522,
        .2500, -.0540,
        .3000, -.0548,
        .3500, -.0549,
        .4000, -.0541,
        .4500, -.0524,
        .5000, -.0497,
        .5500, -.0455,
        .5750, -.0426,
        .6000, -.0389,
        .6250, -.0342,
        .6500, -.0282,
        .6750, -.0215,
        .7000, -.0149,
        .7250, -.0090,
        .7500, -.0036,
        .7750, .0012,
        .8000, .0053,
        .8250, .0088,
        .8500, .0114,
        .8750, .0132,
        .9000, .0138,
        .9250, .0131,
        .9500, .0106,
        .9750, .0060,
        1.000, -.0013,
    ])
    points_2 = points_2.reshape(int(0.5*len(points_2)), 2)[::-1]
    points_2 = points_2.reshape(2*len(points_2))

    points = np.append(points_1, points_2)
    segments = np.empty((int(0.5*len(points)), 2), dtype=np.int64)
    segments[0:-1] = np.array([[i, i+1] for i in np.arange(int(0.5*len(points))-1)])
    segments[-1] = [int(0.5*len(points))-1, 0]
    points = np.append(
        3*points,
        5*np.array([
            -1, -1,
            2, -1,
            2, 1,
            -1, 1,
        ])
    )
    # points *= 3
    segments = np.append(
        segments,
        [
            [int(0.5*len(points))-1-3, int(0.5*len(points))-1-2],
            [int(0.5*len(points))-1-2, int(0.5*len(points))-1-1],
            [int(0.5*len(points))-1-1, int(0.5*len(points))-1-0],
            [int(0.5*len(points))-1-0, int(0.5*len(points))-1-3]
        ],
        axis=0
    )
    insertion_points = np.array([0.0075, 0])
    # points = 2*np.array([[np.cos(2*np.pi*i/20), np.sin(2*np.pi*i/20)] for i in range(20)])
    # points = np.append(
    #     points,
    #     3*np.array([
    #         [-2, -2],
    #         [4, -2],
    #         [4, 2],
    #         [-2, 2]
    #     ]),
    #     axis=0
    # )
    # segments = np.empty((len(points), 2), dtype=np.int64)
    # segments[0:-4-1] = np.array([[i, i+1] for i in range(len(points)-4-1)])
    # segments[-4-1] = [len(points)-5, 0]
    # segments[-4:] = np.array([
    #     [len(points)-4, len(points)-4+1],
    #     [len(points)-4+1, len(points)-4+2],
    #     [len(points)-4+2, len(points)-4+3],
    #     [len(points)-4+3, len(points)-4],
    # ])
    # points = points.reshape(2*len(points))
    # insertion_points = np.array([0, 0])
    return points, segments, insertion_points

if __name__ == "__main__":
    #import sys
    #perf(int(sys.argv[1]))

    points, segments, insertion_points = make_data()

    RDT = RefinedDelaunay(
        points,
        segments,
        insertion_points
    )

    RDT.plotDT()