# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# A test of Pythons numpy library
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# This program was created on Sat Apr 02 2022 by Felix Bo Caspersen (s183319), Mathematics and Technology - DTU

import numpy as np

bla = np.random.random((2,4))
blaBla = np.array([[69, 42], [420, 169]])
blaBlaBla = np.concatenate((bla, blaBla), axis=1)
print("The matrix we wanted:\n", blaBlaBla.tolist())