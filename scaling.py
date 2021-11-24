import sys
import numpy as np

# sys.stdin = open('inp.txt', 'r')
# sys.stdout = open('out.txt', 'w')

print("problem type(1 if max, 0 if min) and no.of variables:")
pb_type, num_var = [int(x) for x in input().split()]

print("no.of <=, =, >= type equations:")
eq_l, eq_e, eq_g = [int(x) for x in input().split()]

eq_total = eq_l + eq_e + eq_g

print("coefficients of objective function(no constant):")
list_obj = [float(x) for x in input().split()]

list_mat = []


print(f"give the matrix A(size {eq_total}*{num_var}):")
for i in range(eq_total):
    list_mat.append([float(x) for x in input().split()])

print(f"give the constants of contraints(size {eq_total}):")
list_cons = [float(x) for x in input().split()]


if pb_type == 0:
	list_obj = [-x for x in list_obj]

mat_main = np.zeros((eq_total, num_var + eq_total + eq_g), dtype=np.float64)


for i in range(eq_total):
	mat_main[i][:num_var] = list_mat[i]
	if i < eq_l:
		mat_main[i][num_var:num_var+eq_l] = np.identity(eq_l)[i]
	elif i < eq_l+eq_e:
		mat_main[i][num_var+eq_l:num_var+eq_l+eq_e] = np.identity(eq_e)[i-eq_l]
	else:
		mat_main[i][num_var+eq_l+eq_e:num_var+eq_total+eq_g] =  list(np.identity(eq_g)[i-eq_l-eq_e]) + list(np.multiply(np.identity(eq_g)[i-eq_l-eq_e], -1))

M = -1e6

A = mat_main.copy()
C = np.zeros((num_var+eq_total+eq_g, 1))
C[:num_var, 0] = list_obj
for i in range(eq_e+eq_g):
	C[i+num_var+eq_l][0] = M

trial_sol = np.zeros((num_var+eq_total+eq_g, 1))
start_val = 1e-3

trial_sol[:num_var, 0] = [start_val]*num_var
trial_sol[num_var+eq_l+eq_e+eq_g:, 0] = [start_val]*eq_g
for i in range(eq_l+eq_e+eq_g):
	trial_sol[num_var+i][0] = list_cons[i]
	for j in range(num_var):
		trial_sol[num_var+i][0] -= trial_sol[j][0]*A[i][j]
	if i >= eq_l+eq_e:
		trial_sol[num_var+i][0] += trial_sol[num_var+eq_g+i][0]

round_off = 4
alpha = 0.5
epsilon = 1e-5
Xi = trial_sol.copy()
iter_cnt = 0

print("\n\n\n")

while True:
	iter_cnt += 1
	D = np.diag(Xi[:, 0])
	A_tilda = np.linalg.multi_dot([A, D])
	C_tilda = np.linalg.multi_dot([D, C])
	A_tilda_T = np.transpose(A_tilda)
	P = np.identity((num_var+eq_total+eq_g)) - np.linalg.multi_dot([A_tilda_T, np.linalg.inv(np.linalg.multi_dot([A_tilda, A_tilda_T])), A_tilda])


	Cp = np.linalg.multi_dot([P, C_tilda])

	v = max([abs(x) for x in Cp[:, 0] if x < 0])

	X_tilda = np.ones((num_var+eq_total+eq_g, 1)) + np.multiply(Cp, alpha/v)

	Xi_plus_1 = np.linalg.multi_dot([D, X_tilda])
	norm = np.linalg.norm(Xi_plus_1[:num_var, 0] - Xi[:num_var, 0])
	if norm <= epsilon:
		print("iteration count: {}".format(iter_cnt))
		print(Xi_plus_1)
		print(f"varible values: {', '.join([str(round(x, round_off)) for x in Xi_plus_1[:num_var, 0]])}")
		answer = np.linalg.multi_dot([Xi_plus_1[:num_var, 0], list_obj])
		print(f"answer: {round(answer if pb_type else -answer, round_off)}")
		break
	Xi = Xi_plus_1