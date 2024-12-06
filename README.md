# 10708-unlearning
Course Project for 10708 (Probabilistic Graphical Models): Unlearning in Diffusion Models


## Commands for running:


### Finetune:
```bash
python finetune.py --erase_concept "parachute" --train_method "xattn" --iterations 100 --lr 1e-5
```

### Generate images:
```bash
python generate.py --erase_concept "parachute" --train_method "xattn" --output_dir tmp --finetuner --model_dir "models_new" --epochs 20
```