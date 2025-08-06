# Add 400ms latency to all satellite containers
docker ps --filter "name=satellite" --format "{{.Names}}" | while read container_name
do
    echo "Injecting latency to $container_name (walker_star)..."
    docker run -d --rm \
        --net container:$container_name \
        gaiaadm/pumba:latest \
        netem --duration 5m delay --time 400
done
