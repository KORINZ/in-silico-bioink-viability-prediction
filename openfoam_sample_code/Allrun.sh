#!/bin/bash

renumberMesh -overwrite

decomposePar -force

# Run mpirun with logging
mpirun -np 6 foamRun -solver incompressibleFluid -parallel > >(tee -a foam.log) 2> >(tee -a error.log >&2)

# Run the subsequent commands
reconstructPar > >(tee -a ReconstructPar.log) 2>&1

rm -r processor*

foamLog foam.log

# Find the latest time directory
latest_time=$(ls -v postProcessing/graphUniform | tail -n 1)

# Calculate average velocity in the y-direction (positive)
awk 'NR==1 {next}
     {
         u_y = sqrt($(NF-1)^2)  # Take the absolute value of velocity
         sum_u_y += u_y
         count++
     }
     END {
         avg_u_y = sum_u_y / count
         printf "Average Velocity (U_y, m/s): %.4f\n", avg_u_y
     }' postProcessing/graphUniform/"$latest_time"/line.xy > data.log

# Read the last row of the "p" file, extract the last value of Probe 0, and multiply by 10
awk 'NR>3 {last_row=$0}
     END {
         split(last_row, values)
         probe0_last_value = values[2] * 10  # Multiply pressure by 10
         printf "Pressure (Probe 0, kPa): %.3f\n", probe0_last_value
     }' postProcessing/probes/0/p >> data.log

# Read the last row of the "shearStress" file and extract the shear stress values for Probe 1
awk 'NR>3 {last_row=$0}
     END {
         match(last_row, /\(([^)]+)\)\s*$/, probe1_stress)
         split(probe1_stress[1], values, /[[:space:]]+/)
         ss_xx = values[1]; ss_xy = values[2]; ss_xz = values[3]
         ss_yy = values[4]; ss_yz = values[5]; ss_zz = values[6]
         shear_stress = sqrt(ss_xx^2 + ss_xy^2 + ss_xz^2 + ss_yy^2 + ss_yz^2 + ss_zz^2)
         printf "Shear Stress (Probe 1, kPa): %.3f\n", shear_stress
     }' postProcessing/probes/0/shearStress >> data.log

# Calculate and record the residence time
awk 'NR==1 {
         avg_u_y = $5
         residence_time = int(0.02 / avg_u_y * 1000)  # Convert to ms and round to integer
         printf "Residence Time (ms): %d\n", residence_time
     }' data.log >> data.log

echo

cat data.log
