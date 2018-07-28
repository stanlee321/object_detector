#!/usr/bin/env python2.7

# test_dnnmodel.py
import argparse
from ownLibraries.object_detection import model_cardetector

parser = argparse.ArgumentParser(description='Add folder to process')
parser.add_argument('-f', '--checkImage', default = None, type=str, help="Add path to the folder to check")

args = parser.parse_args()

if args.checkImage != None:
    rutaDeTrabajo = args.checkImage
    print('Ruta a limpiar: {}'.format(rutaDeTrabajo))
else:
    print('No se introdujo folder a revisar')


def imagenes(rutaDeTrabajo):
    detector = model_cardetector.CarDetector()

    #path = '/home/stanlee321/object_test/2018-02-19_08-54-10_1wm.jpg'
    detection = detector.get_objects(target_image=rutaDeTrabajo)
    print('DETECTION ARE', detection)


if __name__ == '__main__':
    imagenes(rutaDeTrabajo)