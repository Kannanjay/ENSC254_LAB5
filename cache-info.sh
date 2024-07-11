#!/bin/bash

# Set default CPU to cpu0, override if an argument is provided
CPU="cpu0"
if [ $# -eq 1 ]; then
  CPU="cpu$1"
else
  echo "Using default CPU: cpu0"
  echo "To specify another CPU, use the script like this: $0 <cpu_number>"
  echo
fi

for i in /sys/devices/system/cpu/${CPU}/cache/index*; do
    echo "Cache Level: $(cat $i/level)";
    echo "Cache Type: $(cat $i/type)";
    echo "Size: $(cat $i/size)";
        echo "Associativity: $(cat $i/ways_of_associativity)";
    echo "# of Sets: $(cat $i/number_of_sets)";
    echo "Shared CPUs: $(cat $i/shared_cpu_list)";
    echo;
done