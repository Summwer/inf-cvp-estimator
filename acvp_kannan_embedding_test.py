from g6k.utils.util import load_svpchallenge_and_randomize
from random import randint
from fpylll.util import gaussian_heuristic
from g6k.siever import Siever
from math import sqrt
from fpylll import IntegerMatrix, LLL
from g6k.algorithms.pump import pump
from g6k.utils.stats import dummy_tracer
from db import short_vector_sampling_procedure, store_db
import os

def load_cvp_instance(n):
    A, _ = load_svpchallenge_and_randomize(n)
    target_vector = [randint(-2**10,2**10) for _ in range(n)]
    
    return A, target_vector



def acvp_kannan_embedding(A, target_vector):
    g6k = Siever(A,None)
    n = g6k.full_n
    rr = [g6k.M.get_r(i, i) for i in range(g6k.full_n)]
    gh = sqrt(gaussian_heuristic(rr))
    M = round(1.24* gh/sqrt(60))
    
    B = IntegerMatrix(n+1,n+1)
    for i in range(n):
        for j in range(n):
            B[i, j] = A[i, j]
        B[-1, i] = target_vector[i]
    B[-1, -1] = M
  
    B = LLL.reduction(B)
    
    

    return B,M


def acvp_kannnan_embedding_test(n):
    print("=====================")
    print("Start test acvp kannan embeeding with lattice dimension = %d" %n)
    A, target_vector = load_cvp_instance(n)
    B,M = acvp_kannan_embedding(A, target_vector)
    db = short_vector_sampling_procedure(B)
    dbfolder_name = 'db'
    os.makedirs(dbfolder_name, exist_ok=True)
    dbfile_path = os.path.join(dbfolder_name, f'db_%d.txt' %n)
    store_db(db,dbfile_path)
    
    
    g6k = Siever(A,None)
    rr = [g6k.M.get_r(i, i) for i in range(g6k.full_n)]
    gh = sqrt(gaussian_heuristic(rr))
    bound = sqrt(4./3.) * gh

    num = 0
    db_gh_bound_num = 0
    M_db_gh_boung_num = 0
    for v in db:
        norm_v = sqrt(sum([v[0,i]**2 for i in range(g6k.full_n)]))
        if(abs(v[0,-1]) == M):
            num+=1
            if(norm_v<= bound):
                M_db_gh_boung_num += 1
        if(norm_v <= bound):
            db_gh_bound_num += 1
    print("full_count: M = %d, num(v[-1] =+/- M) = %d, db_size = %d, prob = %.4f "  %(M, num, len(db), num/len(db))) 
    print("Restrict norm(v) <= sqrt(4/3) gh: M = %d, num(v[-1] =+/- M) = %d, db_size = %d, prob = %.4f "  %(M, M_db_gh_boung_num, db_gh_bound_num, M_db_gh_boung_num/db_gh_bound_num)) 
    print("-------------------end------------------------")
    


for n in range(60,71):
    acvp_kannnan_embedding_test(n)
