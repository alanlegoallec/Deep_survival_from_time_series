#!/bin/bash
targets=( "Age" "Survival" )
#targets=( "Age" )
algorithms=( "ElasticNet" "GBM" "NeuralNetwork" "CNN" )
memory=8G
for target in "${targets[@]}"; do
	for algorithm in "${algorithms[@]}"; do
		if [ $algorithm == "CNN" ]; then
			predictors=( "demographics+PWA" "all+PWA" "features+PWA" )
		else
			predictors=( "demographics" "features" "PWA" "all" )
		fi
		for predictor in "${predictors[@]}"; do
			partition="short"
			if [ $algorithm == "ElasticNet" ]; then
				time=600
			elif [ $algorithm == "GBM" ]; then
				if [ $target == "Age" ]; then
					time=300
				elif [ $target == "Survival" ]; then
					time=1500
				fi
			elif [ $algorithm == "NeuralNetwork" ]; then
				time=4000
			elif [ $algorithm == "CNN" ]; then
				partition="gpu"
				time=1800
			fi
			if [ $partition == "short" ]; then
				if (( $time > 720 )); then
					partition="medium"
				fi
			fi
			version=PWA_${target}_${predictor}_${algorithm}
			job_name="$version.job"
			out_file="../eo/$version.out"
			err_file="../eo/$version.err"
			args=( -p $partition --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -t $time )
			if [ $partition == "gpu" ]; then
				args+=(
					--gres=gpu:1
					-x compute-g-16-175,compute-g-16-176,compute-g-16-197
				)
			fi
			if ! test -f "$out_file" || ( ! grep -q "Done." "$out_file" ); then
				if [ $(sacct -u al311 --format=JobID,JobName%150,MaxRSS,NNodes,Elapsed,State | grep $version | egrep 'PENDING|RUNNING' | wc -l) -eq 0 ]; then
					echo $version
					sbatch "${args[@]}" Training.sh $target $predictor $algorithm
				fi
			fi
		done
	done
done

