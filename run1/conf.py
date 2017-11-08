
# surface configuration
surface_radius = 50  # radius of the surface hexagon (in fcc-fcc unit)
nn_radius = 2.5  # radius to use for NN (2.5 is fine)
surface_db = 'sdb'

# run configuration
use_image_process = True

# montecarlo
lap_max = 100
steps_per_lap = 500  # number of step per lap. 1 step = + 1 atom on the surface
moves_per_step = 5
target_coverage = 0.95 / 6  # max coverage 1/6 = FCC reconstruction

# temperature
init = 200
final = 10

temperature = lambda lap: init - (init-final) * lap / lap_max

# potential
pot_type = 'NN'
pot_conf =[
     0.100,  # nn 1 ( x3 /  0: 3 ) fcc -> hcp
     0.060,  # nn 2 ( x6 /  3: 9 ) fcc -> fcc
    -0.010,  # nn 3 ( x3 /  9:12 ) fcc -> hcp
    -0.012,  # nn 4 ( x6 / 12:18 ) fcc -> hcp
    -0.020,  # nn 5 ( x6 / 18:24 ) fcc -> fcc (sqrt(3) x sqrt(3) R30)
    -0.015,  # nn 6 ( x6 / 24:30 ) fcc -> fcc
    -0.008,  # nn 7 ( x6 / 30:36 ) fcc -> hcp
    -0.005,  # nn 8 ( x3 / 36:39 ) fcc -> hcp
    #-0.004,  # nn 9   fcc -> hcp
    # nn 10+ ==> energy = 0
    ]

sites_energy_conf = {
    #'fcc': -1.803,
    #'hcp': -1.790,
    'fcc': -0.020,
    'hcp': -0.010,
}
