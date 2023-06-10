#!/bin/bash

max_attempts=5
sleep_time=10 # sleep for 60 seconds between attempts

# Loop through instance names ocd3-2, ocd3-3, ocd3-4, ocd3-5
for instance in od3-2 od3-5; do
  attempt=0

  while true; do
    attempt=$((attempt + 1))

    # Run the gcloud command
    gcloud compute tpus tpu-vm create "$instance" --zone=us-central1-a --accelerator-type=v3-8 --version=tpu-vm-pt-2.0

    # Check if the command was successful
    if [ $? -eq 0 ]; then
      echo "TPU creation for $instance succeeded!"
      break
    else
      echo "TPU creation for $instance failed. Attempt $attempt/$max_attempts."
      if [ $attempt -ge $max_attempts ]; then
        echo "Max attempts reached for $instance. Moving on to the next instance."
        break
      fi

      # Sleep before retrying
      echo "Retrying in $sleep_time seconds..."
      sleep $sleep_time
    fi
  done
done
