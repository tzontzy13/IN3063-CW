from start import start as start_main
from task3_sgd import start as start_pytorch_sgd
from task3_cnn import start as start_pytorch_cnn
import time

start = time.time()

# start_main()
# start_pytorch_sgd()
start_pytorch_cnn()

end = time.time()
print("/n")
print("Elapsed time: {:.2f} seconds".format(end - start))
