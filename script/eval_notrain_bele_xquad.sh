DEVICE=7

model_lang_codes=('ar' 'de' 'ru' 'es' 'zh' 'hi' 'te' 'vi' 'bn')
langs=('arb-Arab' 'deu-Latn' 'rus-Cyrl' 'spa-Latn' 'zho-Hans' 'hin-Deva' 'tel-Telu' 'vie-Latn' 'ben-Beng')
MODEL_NAME=bge-m3
CKPT_PATH=BAAI/bge-m3


for i in ${!model_lang_codes[@]}; do
   model_lang_code=${model_lang_codes[$i]}
   model_lang=${langs[$i]}

   if [[ "$model_lang_code" != "hi" && "$model_lang_code" != "te" && "$model_lang_code" != "vi" && "$model_lang_code" != "bn" ]]; then
   ################################################ XQUAD ################################################

   ## XQUAD en-en ##
   CUDA_VISIBLE_DEVICES=$DEVICE mteb run -m $CKPT_PATH \
      --tasks XQuADRetrieval \
      --languages eng-Latn \
      --output_folder eval_results/xquad/$MODEL_NAME/$model_lang_code/en-en \
      --folder_name $MODEL_NAME \
      --verbosity 3 \
      --overwrite \
      --batch_size 32


   ## XQUAD Lang-Lang ##
   CUDA_VISIBLE_DEVICES=$DEVICE mteb run -m $CKPT_PATH \
      --tasks XQuADRetrieval \
      --languages $model_lang \
      --output_folder eval_results/xquad/$MODEL_NAME/$model_lang_code/$model_lang_code-$model_lang_code \
      --folder_name $MODEL_NAME \
      --verbosity 3 \
      --overwrite \
      --batch_size 32


   ## XQUAD en-Lang ##
   CUDA_VISIBLE_DEVICES=$DEVICE mteb run -m $CKPT_PATH \
      --tasks XQuADCrossRetrieval_EN_LANG \
      --languages $model_lang \
      --output_folder eval_results/xquad/$MODEL_NAME/$model_lang_code/en-$model_lang_code \
      --folder_name $MODEL_NAME \
      --verbosity 3 \
      --overwrite \
      --batch_size 32


   ## XQUAD Lang-en ##
   CUDA_VISIBLE_DEVICES=$DEVICE mteb run -m $CKPT_PATH \
      --tasks XQuADCrossRetrieval_LANG_EN \
      --languages $model_lang \
      --output_folder ../eval_results/xquad/$MODEL_NAME/$model_lang_code/$model_lang_code-en \
      --folder_name $MODEL_NAME \
      --verbosity 3 \
      --overwrite \
      --batch_size 32

   fi

   ############################################### Belebele ################################################

   ## Belebele Eng-Eng ##
   CUDA_VISIBLE_DEVICES=$DEVICE mteb run -m $CKPT_PATH \
      --tasks BelebeleRetrieval \
      --languages eng-Latn-eng-Latn \
      --output_folder eval_results/belebele/$MODEL_NAME/en \
      --folder_name $MODEL_NAME \
      --verbosity 3 \
      --batch_size 32


   ## Belebele Lang-Lang ##
   CUDA_VISIBLE_DEVICES=$DEVICE mteb run -m $CKPT_PATH \
      --tasks BelebeleRetrieval \
      --languages $model_lang \
      --output_folder eval_results/belebele/$MODEL_NAME/$model_lang_code \
      --folder_name $MODEL_NAME \
      --verbosity 3 \
      --batch_size 32


done