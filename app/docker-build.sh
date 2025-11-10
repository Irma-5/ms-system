#!/bin/bash

docker-compose build storage
docker-compose build collector
docker-compose build mlservice
docker-compose build visualization
docker-compose build webmaster

echo "to start services run ./docker-start.sh"
