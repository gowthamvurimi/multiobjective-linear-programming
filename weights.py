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

list_weights = []
print(f"give weight vector(size 1*{num_var}):")
list_weights = [float(x) for x in input().split()]

sum_weights = sum(list_weights)
list_weights = [x/sum_weights for x in list_weights]

print("normalized weight vector is: ", end = "")
print(list_weights)

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
    if list_pb_type[i] == 0:
        list_obj_mat[i] = [-x for x in list_obj_mat[i]]

list_obj = [0.0]*num_var
for i in range(num_obj):
    list_obj = [x+list_weights[i]*y for x, y in zip(list_obj, list_obj_mat[i])]

variables = method.calci(1, num_var, list_obj, eq_l, eq_e, eq_g, list_mat, list_cons)


print("\n\n\n")
print(f"a pareto optimal sol: {[round(x, round_off) for x in variables]}")

for i in range(num_obj):
    print(f"{i+1}th obj val({'max' if list_pb_type[i]==1 else 'min'}): ", end="")
    answer = np.linalg.multi_dot([variables, list_obj_mat[i]])
    print(f"{round(answer if list_pb_type[i] else -answer, round_off)} [individual = {list_indi_ans[i]}, when values {list_indi_var[i]}]")