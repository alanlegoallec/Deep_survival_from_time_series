import sys
from PWA_Class import Postprocessing

self = Postprocessing()
self.compute_performances()
self.generate_barplots()
self.save_data()

# Exit
print('Done.')
sys.exit(0)

