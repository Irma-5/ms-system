#!/bin/bash
if [ -z "$1" ]; then
    echo "Usage: $0 <service_name>"
    exit 1
fi

if [ "$1" == "all" ]; then
    docker-compose logs -f
else
    docker-compose logs -f "$1"
fi
