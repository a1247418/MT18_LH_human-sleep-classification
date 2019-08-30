#!/cluster/apps/sfos/bin/bash
# first argument: flags
# second argument: folder
# 3rd argument: data

# hidden sizes
sizes=("16" "32" "64" "128" "256")
for s in "${sizes[@]}"
do
    ./exp.sh "ms.hidden_size=${s} ms.filter_size=${s}" "caro${s}"
done
#for s in "${sizes[@]}"
#do
#    ./exp.sh "ms.hidden_size=${s}" "caroH${s}"
#done
#for s in "${sizes[@]}"
#do
#    ./exp.sh "ms.filter_size=${s}" "caroF${s}"
#done

# neighbours
#hist=("0" "2" "4" "8" "16")
#for h in "${hist[@]}"
#do
#    ./exp.sh "ds.nbrs=${h} ms.label_nbrs=False" "caroN${h}"
#done

#neighbours + labelling
#for h in "${hist[@]}"
#do
#    ./exp.sh "ds.nbrs=${h} ms.label_nbrs=True" "caroNbp${h}"
#done

#KL
weight=("0" "0.5" "1" "10" "50" "100")
for w in "${weight[@]}"
do
    ./exp.sh "ms.kl_weight=${w}" "caroKL${w}"
done

#Sep channesl
./exp.sh "ms.sep_channels=True" "caroSep"

#LR
lr=("0.0001" "0.001" "0.005" "0.01")
for l in "${lr[@]}"
do
    ./exp.sh "ms.optim='adam,lr=${l}'" "caroLR${l}"
done

#LR
do=("0.00001" "0.2" "0.4" "0.6")
for d in "${do[@]}"
do
    ./exp.sh "ms.dropout=${d}" "caroDO${d}"
done




