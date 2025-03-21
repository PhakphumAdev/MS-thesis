#save cache files in scratch since we have bigger space here
export HF_HOME=$SCRATCH/hf_temp
#make sure you are logged in hf
huggingface-cli whoami
echo $HF_HOME

lm_eval --model hf --model_args pretrained=pakphum/llama3.2-typhoon-passthrough --task fivecommonsense --output_path $SCRATCH/MS-thesis/pakphum/llama3.2-typhoon-passthrough --log_samples




