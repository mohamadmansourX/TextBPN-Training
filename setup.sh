declare -A arr
arr+=( ["CTW1500"]="1A2s3FonXq4dHhD64A2NCWc8NQWMH2NFR" ["TD500"]="1ByluLnyd8-Ltjo9AC-1m7omZnI-FA1u0" ["Total-Text"]="17_7T_-2Bu3KSSg2OkXeCxj97TBsjvueC")
declare -A arpath
arpath+=(["CTW1500"]="ctw1500_data.zip" ["TD500"]="TD500_data.zip" ["Total-Text"]="total-text-mat_data.zip")
declare -A arr2
arr2+=( ["MSRA-TD500-model"]="13ZrWfpv2dJqdv0cDf-hE0n5JQegNY0-i" ["CTW-1500-model"]="13ZrWfpv2dJqdv0cDf-hE0n5JQegNY0-i" ["Total-Text-model"]="13ZrWfpv2dJqdv0cDf-hE0n5JQegNY0-i")
echo "Dataset Downloading" $1
gdown --id  ${arr[$1]}
echo "Pretrained Model Downloading" $2
gdown --id  ${arr[$2]}
unzip -oq ${arpath[$1]} -d data/
unzip -oq ICCV2021_model.zip
