#!/bin/bash

wget http://maxwell.cs.umass.edu/splatnet-data/model_shapenet3d.tar
tar xf model_shapenet3d.tar
mv model_shapenet3d/* .
rm -rf model_shapenet3d
rm model_shapenet3d.tar
