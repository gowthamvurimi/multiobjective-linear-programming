import numpy as np


def calci(pb_type, num_var, list_obj, eq_l, eq_e, eq_g, list_mat, list_cons):
    eq_total = eq_l + eq_e + eq_g
    eq_total_after = eq_l + 2*eq_e + eq_g
    k = num_var + eq_total_after

    if pb_type == 0:
        list_obj = [-x for x in list_obj]
        
    mat_main = np.zeros((k+1, 2*k+1), dtype=np.float64)

    for i in range(eq_l):
        mat_main[i][:num_var+eq_l] = list_mat[i] + list(np.identity(eq_l)[i])
        mat_main[i][-1] = list_cons[i]

    for i in range(eq_e):
        mat_main[2*i+eq_l][:num_var] = list_mat[i+eq_l]
        mat_main[2*i+1+eq_l][:num_var] = [-x for x in list_mat[i+eq_l]]
        mat_main[2*i+eq_l][num_var+eq_l:num_var+eq_l+2*eq_e] = np.identity(2*eq_e)[2*i]
        mat_main[2*i+1+eq_l][num_var+eq_l:num_var+eq_l+2*eq_e] = np.identity(2*eq_e)[2*i+1]
        mat_main[2*i+eq_l][-1] = list_cons[i+eq_l]
        mat_main[2*i+1+eq_l][-1] = -list_cons[i+eq_l]

    for i in range(eq_g):
        mat_main[i+eq_l+2*eq_e][:num_var] = [-x for x in list_mat[i+eq_l+eq_e]]
        mat_main[i+eq_l+2*eq_e][num_var+eq_l+2*eq_e:num_var+eq_l+2*eq_e+eq_g] = np.identity(eq_g)[i]
        mat_main[i+eq_l+2*eq_e][-1] = -list_cons[i+eq_l+eq_e]

    for i in range(num_var):
        mat_main[eq_total_after+i][k:-1] = list(mat_main[:eq_total_after,i]) + list((np.zeros((num_var, num_var)) - np.identity(num_var))[i])
        mat_main[eq_total_after+i][-1] = list_obj[i]

    mat_main[-1][:num_var] = list_obj
    mat_main[-1][k:-1] = [-x for x in mat_main[:eq_total_after,-1]] + [0]*num_var


    list_interior_pt = [0.1]*(2*k+1)
    list_interior_pt[-1] = 1
    mat_main[:, -1] = [-x for x in mat_main[:, -1]]
    list_lambda_col = []
    for i in range(k+1):
        for j in range(2*k):
            mat_main[i][j] = list_interior_pt[j]*mat_main[i][j]
        list_lambda_col.append(-sum([mat_main[i][j] for j in range(2*k+1)]))


    A = np.c_[mat_main, list_lambda_col]


    var_tot = 2*k+2
    eq_tot = k+1
    C = np.zeros((var_tot, 1))
    C[var_tot-1, 0] = 1
    epsilon = 1e-8

    # now the steps start

    X0 = np.full(shape=(var_tot, 1), fill_value=(1.0/(var_tot)), dtype=np.float64)
    r = 1.0/np.sqrt(var_tot*(var_tot-1))
    alpha = 0.5
    round_off = 4

    Xi = X0
    ratio = 1
    lambda_pre = X0[-1, 0]
    iter_cnt = 0
    while 1:
        D = np.diag(Xi[:, 0])
        B = np.matmul(A, D)
        B = np.r_[B, np.ones((1, var_tot))]
        Bt = np.transpose(B)

        Cp = np.linalg.multi_dot([np.identity((var_tot)) - np.linalg.multi_dot([Bt, np.linalg.inv(np.matmul(B, Bt)), B]), D, C])

        mod_Cp = np.linalg.norm(Cp)
        C_cap = np.divide(Cp, mod_Cp)
        Y = X0 - np.multiply(C_cap, alpha*r)
        Xi_plus_1 = np.divide(np.matmul(D, Y), sum(list(np.matmul(D, Y))))
        lambda_new = np.matmul(np.transpose(C), Xi_plus_1)
        if lambda_pre - lambda_new <= epsilon:
            # print("feasible")
            break

        ratio_new = lambda_new/X0[-1, 0]
        if(ratio_new > ratio):
            print("infeasible")
            break
        ratio = ratio_new
        Xi = Xi_plus_1
        lambda_pre = lambda_new
        iter_cnt += 1

    Xi_plus_1[:-1, 0] = [list_interior_pt[i]*Xi_plus_1[i, 0] for i in range(2*k+1)]
    Xans = np.divide(Xi_plus_1, Xi_plus_1[-2, 0])
    return Xans[:num_var, 0]

