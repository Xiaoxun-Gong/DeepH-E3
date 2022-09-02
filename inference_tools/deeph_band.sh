#!/bin/bash
#PBS -N deeph_band
#PBS -l nodes=1:ppn=24
#PBS -l Qlist=n24

# calculate bands from band_i to band_f
# submit num_tasks tasks simultaneously
# place under same directory with hamiltonians_pred.h5, overlaps.h5

# == basic information ==
fermi_level=-3.9143371779349274
lowest_band=-3.1
num_band=100
band_i=1 #$1
band_f=47 #$2
num_tasks=8 #$3
julia=/home/gongxx/apps/julia-1.5.4/bin/julia
sparse_calc=/home/gongxx/projects/DeepH/e3nn_DeepH/DeepH-E3/inference_tools/sparse_calc.jl
work_dir=`pwd` #/home/gongxx/projects/DeepH/e3nn_DeepH/structrues/0627_deephCompare/3_bg/pred_twist/10-9 #$PBS_O_WORKDIR #"/home/xurz/xiaoxun/deeph/0125_MoS2/4-5-2/"

outfile=out.log #out.$PBS_JOBID

cd $work_dir
date > $outfile


# == create config ==
mkdir $work_dir/band
mkdir $work_dir/band/egval_k

loop=$band_i
while [ $loop -le $band_f ]
do

mkdir ${work_dir}/band/egval_k/${loop}
cat > ${work_dir}/band/egval_k/${loop}/band_config.json <<!
{
    "calc_job": "band",
    "which_k": ${loop},
    "fermi_level": $fermi_level,
    "lowest_band": $lowest_band,
    "max_iter": 600,
    "num_band": $num_band,
    "k_data": ["17 0.0 0.0 0.0 0.5 0.0 0.0 Gamma M", "10 0.5 0.0 0.0 0.33333333 0.33333333 0.0 M K", "20 0.33333333 0.33333333 0.0 0.0 0.0 0.0 K Gamma"] 
}
!

# "k_data": ["15 0.5 0.0 0.0  0.0 0.0 0.0  M G", "15 0.0 0.0 0.0  0.66666667 0.3333333 0.000 G K", "15 0.6666667 0.3333333 0.0  0.5 0.0 0.0  K M"]
# ["20 0.33333333 0.66666667 0.0 0.0 0.0 0.0 K Gamma", "17 0.0 0.0 0.0 0.0 0.5 0.0 Gamma M", "10 0.0 0.5 0.0 0.33333333 0.66666667 0.0 M K"]

let "loop++"

done

cp $work_dir/band/egval_k/$band_i/band_config.json $work_dir/band/get_matrix_config.json
sed -i 's/"band"/"get_matrix"/g' $work_dir/band/get_matrix_config.json

# == create sparse_matrix.jld ==

$julia $sparse_calc --input_dir $work_dir --output_dir $work_dir --config $work_dir/band/get_matrix_config.json >> $outfile 2>&1

# diagonalize matrices

loop=$band_i
i_task=0
while [ $loop -le $band_f ]
do
    if [ `jobs | grep Running | wc -l` -lt $num_tasks ]; then
        outfile1="${work_dir}/band/egval_k/${loop}/out.log"
        date > $outfile1
        $julia $sparse_calc --input_dir $work_dir --output_dir ${work_dir}/band/egval_k/${loop}/ --config ${work_dir}/band/egval_k/${loop}/band_config.json >> $outfile1 2>&1 &
        echo "started task $loop successfully"
        let "loop++"
    else
        sleep 1
    fi
done

wait

# copy openmx.Band
cp $work_dir/band/egval_k/$band_i/openmx.Band $work_dir/band/

echo "all tasks completed" >> $outfile
date >> $outfile
