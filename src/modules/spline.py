from multiprocessing import allow_connection_pickling
import numpy as np

def spline(ti, xi):
	n = len(xi)
	dxi = np.diff(xi)
	dti = np.diff(ti)
	ai = xi[:n-1]
	ci = np.zeros(n-1)
	temp = dxi/dti
	result = 3*np.diff(temp)[:n-3]
	sup_diagonal = dti[:n-4]
	sub_diagonal = dti[:n-4]
	main_diagonal = 2*(dti[:n-3] + dti[1:n-2])
	
	# Thomas's alg
	for i in range(n-4):
		m = sub_diagonal[i]/main_diagonal[i]
		main_diagonal[i+1] -= m*sup_diagonal[i]
		result[i+1] -= m*result[i]
	
	ci[0] = 0
	ci[n-2] = 0
	ci[n-3] = result[n-4]/main_diagonal[n-4]
	for i in range(n-4, 0, -1):
		ci[i] = (result[i-1]-sup_diagonal[i-1]*ci[i+1])/main_diagonal[i-1]

	val = dti[:n-2]
	di = np.diff(ci)/(3*val)
	bi = temp[:n-2] - val * (ci[:n-2]+di*val)
	bi = np.append(bi, bi[-1]+2*ci[-2]*val[-1] + 3*di[-1]*val[-1]**2)
	di = np.append(di, 1/dti[-1]**2 * (temp[-1] - ci[-1]*dti[-1] - bi[-1]))
	
	return ai, bi, ci, di

def eval_spline(t, ti, ai, bi, ci, di):
	n = len(ti)
	if t < ti[0] or t > ti[n-1]:
		raise ValueError
	else:
		for i in range(n-1):
			if ti[i] <= t and ti[i+1] >= t:
				d = t - ti[i]
				return ai[i] + d * (bi[i] + d * (ci[i] + d * di[i]))


def new_spline(ti, xi, vi):
	n = len(ti)
	ai = xi[:n-1]
	bi = vi[:n-1]
	dti = np.diff(ti)
	dxi = np.diff(xi)
	dvi = np.diff(vi)
	idti = 1/dti
	
	alphai = dxi - bi * dti
	
	sub_diagonal = -dti[:n-4]
	sup_diagonal = sub_diagonal
	main_diagonal = 3*dti[:n-3]+3*dti[1:n-2]
	
	result = 20/dti[:n-3]**2*alphai[:n-3] - 14/dti[:n-3]*dvi[:n-3] + 20/dti[1:n-2]**2*alphai[1:n-2] - 6/dti[1:n-2]*dvi[1:n-2]
	
	# Thomas's alg
	for i in range(n-4):
		m = sub_diagonal[i]/main_diagonal[i]
		main_diagonal[i+1] -= m*sup_diagonal[i]
		result[i+1] -= m*result[i]
	
	ci = np.zeros(n-1)
	di = np.zeros(n-1)
	ei = np.zeros(n-1)
	gi = np.zeros(n-1)
	
	di[0] = 0
	di[n-2] = 0
	di[n-3] = result[n-4]/main_diagonal[n-4]
	for i in range(n-4, 0, -1):
		di[i] = (result[i-1]-sup_diagonal[i-1]*di[i+1])/main_diagonal[i-1]
				
	for i in range(n-2):
		ci[i+1] = idti[i]*(-5/2*idti[i]*alphai[i]+7/4*dvi[i])+dti[i]/8*(3*di[i+1]-di[i])
	
	ci[0] = 3*ci[1]-dti[0]*di[1]+idti[0]*(10*idti[0]*alphai[0]-6*dvi[0])
	
	for i in range(n-2):
		ei[i] = idti[i]*(-3*di[i+1]+ idti[i]*(7*ci[i+1] + idti[i]*(-11*dvi[i] + 15*idti[i]*alphai[i])))
		gi[i] = idti[i]**2*(di[i+1] + idti[i]*(-2*ci[i+1] + idti[i]*(3*dvi[i] - 4*idti[i]*alphai[i])))
			
	ei[n-2] = idti[n-2]*(-2*di[n-2]+idti[n-2]*(-3*ci[n-2]+idti[n-2]*(-dvi[n-2]+idti[n-2]*5*alphai[n-2])))
	gi[n-2] = idti[n-2]**2*(di[n-2]+idti[n-2]*(2*ci[n-2]+idti[n-2]*(dvi[n-2]-idti[n-2]*4*alphai[n-2])))
	
	return ai, bi, ci, di, ei, gi
		
def eval_new_spline(t, ti, ai, bi, ci, di, ei, gi):
	n = len(ti)
	if t < ti[0] or t > ti[n-1]:
		raise ValueError
	else:
		for i in range(n-1):
			if ti[i] <= t and ti[i+1] > t:
				d = t - ti[i]
				return ai[i] + d * (bi[i] + d * (ci[i] + d * (di[i] + d * (ei[i] + d * gi[i]))))
			if t == ti[n-1]:
				d = t - ti[i]
				return ai[i] + d * (bi[i] + d * (ci[i] + d * (di[i] + d * (ei[i] + d * gi[i]))))

def eval_der_new_spline(t, ti, bi, ci, di, ei, gi):
	n = len(ti)
	if t < ti[0] or t > ti[n-1]:
		raise ValueError
	else:
		for i in range(n-1):
			if ti[i] <= t and ti[i+1] > t:
				d = t - ti[i]
				return bi[i] + d * (2*ci[i] + d * (3*di[i] + d * (4*ei[i] + d * 5*gi[i])))
		if t == ti[n-1]:
			d = t - ti[i]
			return bi[i] + d * (2*ci[i] + d * (3*di[i] + d * (4*ei[i] + d * 5*gi[i])))