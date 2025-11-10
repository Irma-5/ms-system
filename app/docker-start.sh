#!/bin/bash

docker-compose up -d


services=("storage:8003" "collector:8001" "mlservice:8002" "visualization:8004" "webmaster:8000")

for service in "${services[@]}"; do
    name="${service%:*}"
    port="${service#*:}"
    
    if curl -f http://localhost:$port/health > /dev/null 2>&1; then
        echo "$name - healthy"
    else
        echo "$name - no response"
    fi
done

docker-compose ps

echo "To view logs:  docker-compose logs -f [service_name]"
echo "to stop services:  ./docker-stop.sh"
