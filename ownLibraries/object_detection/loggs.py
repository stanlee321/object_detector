import logging

from .paths import DetectionPaths



class DetectionLogs():

    # Set Logging configs
    LOG_FORMAT = "%(levelname)s %(asctime)s - - %(message)s"
    logging.basicConfig(filename=DetectionPaths.LOG_PATH,
                        level=logging.DEBUG,
                        format=LOG_FORMAT)
    logger = logging.getLogger()


    @classmethod
    def info(cls, location, information):
        # Notify that there was not folder
        cls.logger.info('FROM {} :: {}'.format(location, information))

    @classmethod
    def warning(cls, location, warning):
        cls.logger.warning('FROM {} :: {}'.format(location, warning))

