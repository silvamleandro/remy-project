# This is the main script to run the Federated Learning (FL) setup
# It executes the Flower Server (server.py) and multiple clients with the script client.py
#!/bin/bash

echo "[REMY][NETWORK MODULE] Starting Federated Learning server using Flower"

python server.py &
sleep 10  # Sleep to give the server enough time to start

for filename in ./data/*.csv; do
    echo "[REMY][NETWORK MODULE] Starting client with $filename data"
    python client.py --data_path=${filename} &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
