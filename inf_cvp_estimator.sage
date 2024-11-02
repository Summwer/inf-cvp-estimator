#Estimator for approximate SVP and CVP under the infinite norm.
#The gate-count model is taken from [AGPS20], and the progressive BKZ simulator is taken from [DDGR20].

def search_beta_kappa(logvol, dim, min_beta, max_beta, lower_gap, upper_gap, bound, mod='CVP'):
    # main algorithm
    # logvol: logarithm of the volume of lattice; dim: dimension of lattice; min_beta, max_beta: the search range of beta;
    # lower_gap, upper_gap: the search range of kappa is [beta-lower_gap,beta+upper_gap];
    # bound: the bound of vector length in infinite norm for SVP or CVP; mod: 'CVP' or 'SVP';
    # return value: (1) log_2(gates); (2) blocksize of progressive BKZ; (3) dimension of the final sieve;
    delta = compute_delta(max_beta,dim)  # this step takes a lot of time
    min_cost = -1
    for beta in range(min_beta,max_beta):
        for kappa in range(beta-lower_gap,beta+upper_gap):
            if(mod=='CVP'):
                cost = acvp_cost(logvol, dim, beta, delta[beta], kappa, bound)
            else:
                cost = asvp_cost(logvol, dim, beta, delta[beta], kappa, bound)
            if(min_cost < 0 or min_cost > cost):
                min_cost = cost
                min_beta = beta
                min_kappa = kappa
    return (n(min_cost),min_beta,min_kappa)

def prob_restriction(dim, bound, len, mod='CVP'):
    if(mod=='CVP'):
        const = 1+(2/dim)*log(sqrt(2)+1)*log(sqrt(2)+1)
        len = len / sqrt(const)

    T=RealDistribution('gaussian',1)
    t = sqrt(dim)*bound/len
    log_prob = log(T.cum_distribution_function(t)-T.cum_distribution_function(t*-1))*dim
    if(mod=='CVP'):
        return log_prob+log(6-4*sqrt(2))
    else:
        return log_prob

def acvp_cost(logvol, dim, beta, delta, kappa, bound):

    len_fac = sqrt(exp(2*(log_gh(kappa,log_vol(delta,dim+1,0,kappa))+log(4/3)/2))+exp(2*(delta*(dim+1 - 2*(dim+1)*kappa/dim))))
    if(kappa<=beta):
        len_fac = sqrt(exp(2*(log_vol(delta,dim+1,0,1)+log(4/3)/2))+exp(2*(delta*(dim+1 - 2*(dim+1)*kappa/dim))))

    len_prime = calc_len_prime(len_fac,logvol,dim)

    log_prob = prob_restriction(dim,bound,len_prime,'CVP')

    if(log_prob+kappa*log(4/3)/2 <0):
        search_cost = log2(C)+agps20_gates(kappa) - (log_prob+kappa*log(4/3)/2)/log(2)
        return log(2**pro_theo_bkz_cost(dim+1, beta)[0]+2**search_cost)/log(2)

    f_prime = 0
    log_gamma = log(4/3)/2
    len_fac = sqrt(4/3) * exp(log_gh(kappa,log_vol(delta,dim+1,0,kappa)))
    if(kappa<=beta):
        len_fac = sqrt(4/3) * exp(log_vol(delta,dim+1,0,1))
    log_size = kappa*log(4/3)/2
    while(log_size + prob_restriction(dim,bound,calc_len_prime(len_fac,logvol,dim),'CVP')>=0 and log_gamma>0):
        #print(n(log_size), n(calc_len_prime(len_fac,logvol,dim)),n(prob_restriction(dim,bound,calc_len_prime(len_fac,logvol,dim))),n(log_gamma),f_prime)
        f_prime += 1
        log_gamma = log(4/3)/2-delta*(f_prime*(dim+1)/dim)
        len_fac = exp(log_gamma + log_gh(kappa,log_vol(delta,dim+1,0,kappa)))
        if(kappa<=beta):
            len_fac = exp(log_gamma + log_vol(delta,dim+1,0,1))
        log_size = kappa * log_gamma
    f_prime -= 1
    return log(2**pro_theo_bkz_cost(dim+1, beta)[0]+2**(log2(C)+agps20_gates(kappa-f_prime)))/log(2)

def asvp_cost(logvol, dim, beta, delta, kappa, bound):

    len_fac = sqrt(exp(2*(log_gh(kappa,log_vol(delta,dim,0,kappa))+log(4/3)/2))+exp(2*(delta*(dim - 2*dim*kappa/(dim-1)))))
    if(kappa<=beta):
        len_fac = sqrt(exp(2*(log_vol(delta,dim,0,1)+log(4/3)/2))+exp(2*(delta*(dim - 2*dim*kappa/(dim-1)))))

    len_prime = exp(logvol/dim) * len_fac

    log_prob = prob_restriction(dim,bound,len_prime,'SVP')

    if(log_prob+kappa*log(4/3)/2 <0):
        search_cost = log2(C)+agps20_gates(kappa) - (log_prob+kappa*log(4/3)/2)/log(2)
        return log(2**pro_theo_bkz_cost(dim, beta)[0]+2**search_cost)/log(2)

    f_prime = 0
    log_gamma = log(4/3)/2
    len_fac = sqrt(4/3) * exp(log_gh(kappa,log_vol(delta,dim,0,kappa)))
    if(kappa<=beta):
        len_fac = sqrt(4/3) * exp(log_vol(delta,dim,0,1))
    log_size = kappa*log(4/3)/2
    while(log_size + prob_restriction(dim,bound,exp(logvol/dim) * len_fac,'SVP')>=0 and log_gamma>0):
        f_prime += 1
        log_gamma = log(4/3)/2-delta*(f_prime*dim/(dim-1))
        len_fac = exp(log_gamma + log_gh(kappa,log_vol(delta,dim,0,kappa)))
        if(kappa<=beta):
            len_fac = exp(log_gamma + log_vol(delta,dim,0,1))
        log_size = kappa * log_gamma
    f_prime -= 1
    return log(2**pro_theo_bkz_cost(dim, beta)[0]+2**(log2(C)+agps20_gates(kappa-f_prime)))/log(2)

def calc_len_prime(len_fac, logvol, dim):
    const = 1+(2/dim)*log(sqrt(2)+1)*log(sqrt(2)+1)
    return (exp(logvol/(dim+1))* (sqrt(2/dim)*log(sqrt(2)+1)/sqrt(const))**(1/(dim+1)) * len_fac) ** ((dim+1)/dim)







def log_gh(d,logvol):
    return log(d/(2*pi*e))/2+logvol/d

def log_vol(delta,d,i,j):
    return delta*((d-(i+j-1)*d/(d-1))*(j-i))

def log_rhf(beta):
    small = {0: 1e20, 1: 1e20, 2: 1.021900, 3: 1.020807, 4: 1.019713, 5: 1.018620,
             6: 1.018128, 7: 1.017636, 8: 1.017144, 9: 1.016652, 10: 1.016160,
             11: 1.015898, 12: 1.015636, 13: 1.015374, 14: 1.015112, 15: 1.014850,
             16: 1.014720, 17: 1.014590, 18: 1.014460, 19: 1.014330, 20: 1.014200,
             21: 1.014044, 22: 1.013888, 23: 1.013732, 24: 1.013576, 25: 1.013420,
             26: 1.013383, 27: 1.013347, 28: 1.013310, 29: 1.013253, 30: 1.013197,
             31: 1.013140, 32: 1.013084, 33: 1.013027, 34: 1.012970, 35: 1.012914,
             36: 1.012857, 37: 1.012801, 38: 1.012744, 39: 1.012687, 40: 1.012631,
             41: 1.012574, 42: 1.012518, 43: 1.012461, 44: 1.012404, 45: 1.012348,
             46: 1.012291, 47: 1.012235, 48: 1.012178, 49: 1.012121, 50: 1.012065}
    if(beta<50):
        return log(small[beta])
    return (log(beta/(2*pi*e))+log(beta*pi)/beta)/(2*(beta-1))



from mpmath import mp

def log2(x):
    return ln(x)/ln(2.)

agps20_gate_data = {
          64  :42.5948446291284,   72  :44.8735917172503, 80  :47.4653141889341, 88  :50.0329479433691, 96  :52.5817667347844,  
          104 :55.1130237325179,   112 :57.6295421450947, 120 :60.133284108578,  128 :62.1470129451821, 136 :65.4744488064273, 
          144 :67.951405476229,    152 :70.0494944191399, 160 :72.50927387359,   168 :74.9619105412039, 176 :77.4100782579645, 
          184 :79.3495443657483,   192 :81.7856479853679, 200 :84.2178462414349, 208 :86.646452845262,  216 :89.0717383389617, 
          224 :91.4939375786565,   232 :93.9132560751063, 240 :96.3298751307529, 248 :98.7439563146036, 256 :101.155644837658, 
          264 :104.091650357302,   272 :106.500713866161, 280 :108.907671199501, 288 :111.312627066864, 296 :113.715679081585, 
          304 :116.11691871212,    312 :118.516432037545, 320 :120.914300351043, 328 :123.310600632063, 336 :125.705405925853, 
          344 :128.098785623819,   352 :130.490805751072, 360 :132.881529104042, 368 :135.271015458153, 376 :137.659321707881, 
          384 :140.046501985502,   392 :142.432607773479, 400 :144.817688009257, 408 :147.201789183958, 416 :149.584955436701, 
          424 :151.967228645918,   432 :154.348648518547, 440 :156.729252677678, 448 :159.109076748918, 456 :161.488154445581, 
          464 :163.866517652676,   472 :166.24419650959,  480 :168.621219491327, 488 :170.997613488119, 496 :173.373403883249, 
          504 :175.748614628914,   512 :178.123268319974, 520 :180.931640474467, 528 :183.305745118107, 536 :185.679338509895, 
          544 :188.052439374005,   552 :190.425065356218, 560 :192.797233085084, 568 :195.168958230518, 576 :197.540255559816, 
          584 :199.911138991095,   592 :202.281621644196, 600 :204.651715889082, 608 :207.02143339179,  616 :209.390785157985, 
          624 :211.759781574203,   632 :214.128432446848, 640 :216.496747039019, 648 :218.864734105257, 656 :221.232401924303, 
          664 :223.599758329925,   672 :225.96681073994,  680 :228.333566183483, 688 :230.700031326626, 696 :233.066212496418, 
          704 :235.43211570344,    712 :237.797746662944, 720 :240.163110814653, 728 :242.528213341298, 736 :244.893059185964, 
          744 :247.25765306831,    752 :249.621999499728, 760 :251.986102797502, 768 :254.349967098032, 776 :256.71359636917,  
          784 :259.076994421734,   792 :261.440164920231, 800 :263.803111392861, 808 :266.165837240825, 816 :268.528345816343, 
          824 :270.890640143248,   832 :273.252723321704, 840 :275.614598434176, 848 :277.976268306208, 856 :280.337735739304, 
          864 :282.699003457275,   872 :285.060074111424, 880 :287.420950285349, 888 :289.781634499399, 896 :292.142129214795, 
          904 :294.502436837451,   912 :296.862559721505, 920 :299.222500172584, 928 :301.582260450819, 936 :303.941842773632, 
          944 :306.301249318305,   952 :308.660482224348, 960 :311.019543595679, 968 :313.378435502636, 976 :315.737159983825, 
          984 :318.095719047813,   992 :320.454114674691, 1000:322.8123488175,   1008:325.170423403542,1016:327.52834033558, 
          1024:329.886101492934
          }


# Function C from AGPS20 source code
def caps_vol(d, theta, integrate=False, prec=None):
    """
    The probability that some v from the sphere has angle at most theta with some fixed u.

    :param d: We consider spheres of dimension `d-1`
    :param theta: angle in radians
    :param: compute via explicit integration
    :param: precision to use

    EXAMPLE::

        sage: C(80, pi/3)
        mpf('1.0042233739846629e-6')

    """
    prec = prec if prec else mp.prec
    with mp.workprec(prec):
        theta = mp.mpf(theta)
        d = mp.mpf(d)
        if integrate:
            r = (
                1
                / mp.sqrt(mp.pi)
                * mp.gamma(d / 2)
                / mp.gamma((d - 1) / 2)
                * mp.quad(lambda x: mp.sin(x) ** (d - 2), (0, theta), error=True)[0]
            )
        else:
            r = mp.betainc((d - 1) / 2, 1 / 2.0, x2=mp.sin(theta) ** 2, regularized=True) / 2
        return r


# Return log2 of the number of gates for FindAllPairs according to AGPS20
def agps20_gates(beta_prime):
    k = beta_prime / 8
    if k != round(k):
        x = k - floor(k)
        d1 = agps20_gates(8*floor(k))
        d2 = agps20_gates(8*(floor(k) + 1))
        return x * d2 + (1 - x) * d1
    return agps20_gate_data[beta_prime]

# Return log2 of the number of vectors for sieving according to AGPS20
def agps20_vectors(beta_prime):
    k = round(beta_prime)
    N = 1./caps_vol(beta_prime, mp.pi/3.)
    return log2(N)


# Progressivity Overhead Factor
C = 1./(1.- 2**(-.292))



#theo d4f 2
def dims4free(beta):
    return ceil(beta * ln(4./3.) / ln(beta/(2*pi*exp(1))))


#cost of bkz with progressive sieve
def theo_bkz_cost(n, beta,J=1):
    if(beta <=10):
        return (0,0)
    #f = dim4free_wrapper(default_dim4free_fun,beta)
    f = dims4free(beta)
    beta_prime = floor(beta - f)
    if(beta_prime < 64 or beta < beta_prime):
        return (0,0)
    elif(beta_prime > 1024):
        return (float("inf"),float("inf"))
    else:
        gates = log2((1.*(n+2*f-beta)/J)*C) + agps20_gates(beta_prime)
        bits = log2(8*beta_prime) + agps20_vectors(beta_prime)
        return (gates, bits)


#cost of progressive bkz with progressive sieve
def pro_theo_bkz_cost(n, beta,J=1):
    if(beta <=10):
        return (0,0)
    #beta_prime = floor(beta - dim4free_wrapper(default_dim4free_fun,beta))
    beta_prime = floor(beta - dims4free(beta))
    if(beta_prime < 64 or beta < beta_prime):
        return (0,0)
    elif(beta_prime > 1024):
        return (float("inf"),float("inf"))
    else:
        gates = log2((1.*(n-beta)/J)*C*C) + agps20_gates(beta_prime)
        bits = log2(8*beta_prime) + agps20_vectors(beta_prime)
        return (gates, bits)


def theo_pump_cost(beta):
    if(beta <=10):
        return (0,0)
    beta_prime = floor(beta - dims4free(beta))
    if(beta_prime < 64 or beta < beta_prime):
        return (0,0)
    elif(beta_prime > 1024):
        return (float("inf"),float("inf"))
    else:
        gates = log2(C) + agps20_gates(beta_prime)
        bits = log2(8*beta_prime) + agps20_vectors(beta_prime)

        return (gates, bits)

rk = [0.789527997160000, 0.780003183804613, 0.750872218594458, 0.706520454592593, 0.696345241018901, 0.660533841808400, 0.626274718790505, 0.581480717333169, 0.553171463433503, 0.520811087419712, 0.487994338534253, 0.459541470573431, 0.414638319529319, 0.392811729940846, 0.339090376264829, 0.306561491936042, 0.276041187709516, 0.236698863270441, 0.196186341673080, 0.161214212092249, 0.110895134828114, 0.0678261623920553, 0.0272807162335610, -
      0.0234609979600137, -0.0320527224746912, -0.0940331032784437, -0.129109087817554, -0.176965384290173, -0.209405754915959, -0.265867993276493, -0.299031324494802, -0.349338597048432, -0.380428160303508, -0.427399405474537, -0.474944677694975, -0.530140672818150, -0.561625221138784, -0.612008793872032, -0.669011014635905, -0.713766731570930, -0.754041787011810, -0.808609696192079, -0.859933249032210, -0.884479963601658, -0.886666930030433]
simBKZ_c = [None] + [rk[-i] - sum(rk[-i:]) / i for i in range(1, 46)]

pruning_proba = .5
simBKZ_c += [RR(log_gh(d,0) - log(pruning_proba) / d) / log(2.) for d in range(46, 1000)]


def simBKZ(l, beta, tours=1, c=simBKZ_c):

    n = len(l)
    l2 = copy(vector(RR, l))

    for k in range(n - 1):
        d = min(beta, n - k)
        f = k + d
        logV = sum(l2[k:f])
        lma = logV / d + c[d]

        if lma >= l2[k]:
            continue

        diff = l2[k] - lma
        l2[k] -= diff
        for a in range(k + 1, f):
            l2[a] += diff / (f - k - 1)

    return l2

def compute_delta(beta, dim):

    delta_beta = [0.0]*beta
    l = [0.0]*dim
    for i in range(dim):
        l[i] = (log_rhf(2)*(dim - 1 - 2 * i)) / log(2)
    for beta_prime in range(2 , beta):
        l = simBKZ(l, beta_prime, 1)
        delta_beta[beta_prime] = l[0]*log(2)/dim
    return delta_beta
