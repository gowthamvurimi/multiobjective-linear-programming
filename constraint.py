import sys
import numpy as np
import method

# sys.stdin = open('inp.txt', 'r')
# sys.stdout = open('out.txt', 'w')

print("give no.of variables & no.of objective functions:")
num_var, num_obj = [int(x) for x in input().split()]

list_obj_mat = []
print(f"give matrix C(size {num_obj}*{num_var}):")
for i in range(num_obj):
    list_obj_mat.append([float(x) for x in input().split()])

list_pb_type = []
print(f"give pb types(size 1*{num_obj}) of objective functions(1 max, 0 min):")
list_pb_type = [int(x) for x in input().split()]

list_constraints = []
print(f"give constraint vector for objectives (size 1*{num_obj}):")
list_constraints = [float(x) for x in input().split()]


print("no.of <=, =, >= type equations:")
eq_l, eq_e, eq_g = [int(x) for x in input().split()]

eq_total = eq_l + eq_e + eq_g
eq_total_after = eq_l + 2*eq_e + eq_g
k = num_var + eq_total_after


list_mat = []

print(f"give the matrix A(size {eq_total}*{num_var}):")
for i in range(eq_total):
    list_mat.append([float(x) for x in input().split()])

print(f"give the constants of contraints(size {eq_total}):")
list_cons = [float(x) for x in input().split()]


round_off = 4

list_indi_var = []
list_indi_ans = []
for i in range(num_obj):
    vars = method.calci(list_pb_type[i], num_var, list_obj_mat[i], eq_l, eq_e, eq_g, list_mat, list_cons)
    list_indi_var.append([round(x, round_off) for x in vars])
    ans = np.linalg.multi_dot([vars, list_obj_mat[i]])
    list_indi_ans.append(round(ans, round_off))



for i in range(num_obj):
    if list_pb_type[i] == 1:
        list_constraints[i] *= -1
        list_obj_mat[i] = [-x for x in list_obj_mat[i]]



for i in range(num_obj):
    vars = method.calci(0, num_var, list_obj_mat[i], eq_l+num_obj-1, eq_e, eq_g, list_obj_mat[:i] + list_obj_mat[i+1:] + list_mat, list_constraints[:i]+list_constraints[i+1:]+list_cons)
    print(f"\n\n\nwhen {i+1}th is objective function:")
    print(f"variable values: {[round(x, round_off) for x in vars]}")
    for j in range(num_obj):
        answer = np.linalg.multi_dot([vars, list_obj_mat[j]])
        print(f"{round(-answer if list_pb_type[j] else answer, round_off)} [individual = {list_indi_ans[j]}, when values {list_indi_var[j]}]")


