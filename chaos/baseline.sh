# Sequentially inject 200ms latency to pairs of satellites for 5 minutes each

satellites=(satellite_1 satellite_2 satellite_3 satellite_4 satellite_5)

for i in {0..4}
do
    # Select two satellites in a sliding window (wrap around at the end)
    s1=${satellites[$i]}
    s2=${satellites[$(( (i+1)%5 ))]}
    echo "Injecting 200ms latency to $s1 and $s2 for 5 minutes (baseline)..."
    docker run -d --rm --net container:$s1 gaiaadm/pumba:latest netem --duration 5m delay --time 200
    docker run -d --rm --net container:$s2 gaiaadm/pumba:latest netem --duration 5m delay --time 200
    # Write chaos status for both satellites
echo "Chaos injected: 200ms latency for 5m (started at $(date))" > shared_logs/${s1}.chaos.txt
    echo "Chaos injected: 200ms latency for 5m (started at $(date))" > shared_logs/${s2}.chaos.txt
    # Wait 5 minutes before moving to the next pair
    sleep 300
done