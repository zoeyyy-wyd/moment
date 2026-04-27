Still working on it...



.

├── LICENSE

├── README.md

├── commands\_pipeline.txt

├── data

│   ├── LICENSE

│   ├── README.md

│   ├── highlight\_test\_release.jsonl

│   ├── highlight\_test\_with\_gt.jsonl

│   ├── highlight\_train\_release.jsonl

│   ├── highlight\_val\_release.jsonl

│   └── subs\_train.jsonl

├── extract\_features.py

├── features

│   ├── clip\_features

│   ├── clip\_sub\_features

│   │   └── args.json

│   ├── clip\_text\_features

│   │   └── args.json

│   ├── pann\_features

│   └── slowfast\_features

├── inference.sh

├── inference\_sf.py

├── model\_best\_pretrained.ckpt

├── model\_final.ckpt

├── moment\_detr

│   ├── \_\_init\_\_.py

│   ├── config.py

│   ├── inference.py

│   ├── matcher.py

│   ├── misc.py

│   ├── model.py

│   ├── position\_encoding.py

│   ├── postprocessing\_moment\_detr.py

│   ├── scripts

│   │   ├── inference.sh

│   │   ├── pretrain.sh

│   │   └── train.sh

│   ├── span\_utils.py

│   ├── start\_end\_dataset.py

│   ├── text\_encoder.py

│   └── transformer.py

├── pipeline.py

├── pretrain.sh

├── requirements.txt

├── results

│   ├── hl-video\_tef-exp-2026\_04\_25\_19\_19\_47

│   │   ├── best\_hl\_val\_preds.jsonl

│   │   ├── best\_hl\_val\_preds\_metrics.json

│   │   ├── code.zip

│   │   ├── eval.log.txt

│   │   ├── inference\_hl\_val\_None\_preds.jsonl

│   │   ├── inference\_hl\_val\_None\_preds\_metrics.json

│   │   ├── latest\_hl\_val\_preds.jsonl

│   │   ├── latest\_hl\_val\_preds\_metrics.json

│   │   ├── model\_best.ckpt

│   │   ├── model\_e0049.ckpt

│   │   ├── model\_e0099.ckpt

│   │   ├── model\_e0149.ckpt

│   │   ├── model\_e0199.ckpt

│   │   ├── model\_latest.ckpt

│   │   ├── opt.json

│   │   ├── tensorboard\_log

│   │   │   └── events.out.tfevents.1777144789.moment.31958.0

│   │   └── train.log.txt

│   └── hl-video\_tef-pt-2026\_04\_25\_08\_22\_54

│       ├── best\_hl\_val\_preds.jsonl

│       ├── best\_hl\_val\_preds\_metrics.json

│       ├── code.zip

│       ├── eval.log.txt

│       ├── latest\_hl\_val\_preds.jsonl

│       ├── latest\_hl\_val\_preds\_metrics.json

│       ├── model\_best.ckpt

│       ├── model\_e0009.ckpt

│       ├── model\_e0019.ckpt

│       ├── model\_e0029.ckpt

│       ├── model\_e0039.ckpt

│       ├── model\_e0049.ckpt

│       ├── model\_e0059.ckpt

│       ├── model\_e0069.ckpt

│       ├── model\_e0079.ckpt

│       ├── model\_e0089.ckpt

│       ├── model\_e0099.ckpt

│       ├── model\_latest.ckpt

│       ├── opt.json

│       ├── tensorboard\_log

│       │   └── events.out.tfevents.1777105377.moment.10840.0

│       └── train.log.txt

├── run.py

├── run\_on\_video

│   ├── clip

│   │   ├── \_\_init\_\_.py

│   │   ├── bpe\_simple\_vocab\_16e6.txt.gz

│   │   ├── clip.py

│   │   ├── model.py

│   │   └── simple\_tokenizer.py

│   ├── data\_utils.py

│   ├── example

│   │   ├── RoripwjYFp8\_60.0\_210.0.mp4

│   │   └── queries.jsonl

│   ├── model\_utils.py

│   └── moment\_detr\_ckpt

│       ├── README.md

│       ├── eval.log.txt

│       ├── inference\_hl\_val\_test\_code\_preds.jsonl

│       ├── inference\_hl\_val\_test\_code\_preds\_metrics.json

│       ├── model\_best.ckpt

│       ├── opt.json

│       └── train.log.txt

├── standalone\_eval

│   ├── README.md

│   ├── eval.py

│   ├── eval\_sample.sh

│   ├── sample\_val\_preds.jsonl

│   ├── sample\_val\_preds\_metrics\_raw.json

│   └── utils.py

├── train.py

├── train.sh

├── utils

│   ├── basic\_utils.py

│   ├── model\_utils.py

│   ├── temporal\_nms.py

│   ├── tensor\_utils.py

│   └── windows\_utils.py

└── video

&#x20;   └── video.mp4



21 directories, 104 files

