import os

# Own tools
from .tools import DetectionTools

class DetectionPaths():

    home = os.getenv('HOME') + '/'

    path_to_workdir     = home + 'WORKDIR/'

    LOG_FOLDER = path_to_workdir +  'Logs/'

    LOG_PATH = LOG_FOLDER + 'LOGGIN_obj_detection_{}.log'.format(DetectionTools.todaydate())
