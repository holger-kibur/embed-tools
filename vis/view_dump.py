import numpy as np
import re
import matplotlib.pyplot as plt

class ParticleFrame():
    def __init__(self, particles, covar):
        self.particles = particles
        self.covar = covar

def dump_mat_to_np(stream):
    np_string = ""
    for line in stream.readlines():
        if "+" in line:
            break
        np_string += re.sub(r"\s+", " ", re.sub(r"\|", "", line)).strip() + ";"
    return np.matrix(np_string[:-1])

with open("dumpfile.txt", "r") as df:
    while True:
        line = df.readline()
        if "Particles" in line:
            break
    particles = dump_mat_to_np(df)
print(particles[:, 0].A1)
plt.scatter(particles[:, 0].A1, particles[:, 1].A1)
plt.show()
    
