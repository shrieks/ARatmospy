import numpy as np

# create CP profile
def create_compressed_cp_for_cdr(gltype, fatype):

    # the types tell use bad, typical or good for each
    # 0 = bad, 1, = typical, 2 = good

    # basics
    # based on a composite of my making by rougly grouping them together
    #                       r0 (m)  ,vel(m/s), dir (deg), alt(m)
    full_params = np.array([[0.40	,6.9	,284,     0	],
                            [0.78	,7.5	,267,    25	],
                            [1.07	,7.8	,244,    50	],
                            [1.12	,8.3	,267,   100	],
                            [0.84	,9.6	,237,   200	],
                            [0.68	,9.9	,232,   400	],
                            [0.66	,9.6	,286,   800	],
                            [0.91	,10.1	,293,  1600	],
                            [0.40	,7.2	,270,  3400	],
                            [0.50	,16.5	,269,  6000	],
                            [0.85	,23.2	,259,  7600	],
                            [1.09	,32.7	,259, 13300	],
                            [1.08	,5.7	,320, 16000	]])


    r0s  = full_params[:,0] # column 0
    vels = full_params[:,1]
    angles = full_params[:,2]
    alts = full_params[:,3]

    layer1 = [0, 1, 3, 6, 7, 8, 12]
    layer1dom = 0

    layer2 = [2, 4, 5]
    layer2dom = 5

    layer3 = [9, 10, 11]
    layer3dom = 10

    n_layers = 3 # ground and FS
    paracube = np.zeros((n_layers, full_params.shape[-1]))

    paracube[:,0] = np.array([(r0s[layer1]**(-5./3.)).sum()**(-3./5.), 
                              (r0s[layer2]**(-5./3.)).sum()**(-3./5.), 
                              (r0s[layer3]**(-5./3.)).sum()**(-3./5.)])
    paracube[:,1] = np.array([vels[layer1dom], vels[layer2dom], vels[layer3dom]])
    paracube[:,2] = np.array([angles[layer1dom], angles[layer2dom], 
                              angles[layer3dom]])
    paracube[:,3] = np.array([alts[layer1dom], alts[layer2dom], alts[layer3dom]])

    return paracube



# create parameters
def cdr_create_parameters(atmtype):
    
    if atmtype == 0:
        cp_params = create_compressed_cp_for_cdr(0,0)
    else:
        cp_params = create_compressed_cp_for_cdr(0,0)
        cp_params[:,0]  = cp_params[:,0]*atmtype**(-3./5.)
                    
    return cp_params
