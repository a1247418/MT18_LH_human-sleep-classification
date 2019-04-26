n_jobs=300
max_job=1792387

for (( i=0; i<${n_jobs}; i++ ));
do
    bkill $((max_job - i))
done