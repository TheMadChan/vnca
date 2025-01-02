@echo off

@REM ------------------------------- MNIST dataset experiments ---------------------------------------------------------------------

@REM python vae_nca.py --dataset mnist --n_updates 10 --batch_size 32 --test_batch_size 32 --z_size 64 --learning_rate 1e-4 
@REM python vae_nca.py --dataset mnist --n_updates 1000 --batch_size 32 --test_batch_size 32 --z_size 16 --learning_rate 1e-4 
@REM python vae_nca.py --dataset mnist --n_updates 1000 --batch_size 32 --test_batch_size 32 --z_size 64 --learning_rate 1e-4 
@REM python vae_nca.py --dataset mnist --n_updates 1000 --batch_size 32 --test_batch_size 32 --z_size 128 --learning_rate 1e-4 
@REM python vae_nca.py --dataset mnist --n_updates 1000 --batch_size 32 --test_batch_size 32 --z_size 256 --learning_rate 1e-4 
@REM python vae_nca.py --dataset mnist --n_updates 1000 --batch_size 64 --test_batch_size 32 --z_size 64 --learning_rate 1e-4 

@REM ------------------------Fracture dataset (binary threshold tuning) ---------------------------------------------------------------------

@REM python vae_nca.py --n_updates 5000 --bin_threshold 0.5
@REM python generate_fractures.py 

@REM python vae_nca.py --n_updates 5000 --bin_threshold 0.75
@REM python generate_fractures.py 


@REM ----------------------- Fracture dataset (batch size tuning) ---------------------------------------------------------------------

@REM python vae_nca.py --n_updates 5000 --batch_size 64
@REM python generate_fractures.py --n_updates 5000 --batch_size 64
@REM python generate_fractures.py --n_updates 5000 --batch_size 64 --target_image_size 64

@REM python vae_nca.py --n_updates 5000 --batch_size 32
@REM python generate_fractures.py --n_updates 5000 --batch_size 32
@REM python generate_fractures.py --n_updates 5000 --batch_size 32 --target_image_size 64

@REM python vae_nca.py --n_updates 5000 --batch_size 16
@REM python generate_fractures.py --n_updates 5000 --batch_size 16
@REM python generate_fractures.py --n_updates 5000 --batch_size 16 --target_image_size 64


@REM -------------------- Fracture dataset (latent space size tuning) ---------------------------------------------------------------------

@REM python vae_nca.py --n_updates 5000 --z_size 128
@REM python generate_fractures.py 

@REM python vae_nca.py --n_updates 5000 --z_size 64
@REM python generate_fractures.py --n_updates 5000 --z_size 64
@REM python generate_fractures.py --n_updates 5000 --z_size 64 --target_image_size 64

@REM python vae_nca.py --n_updates 5000 --z_size 32
@REM python generate_fractures.py --n_updates 5000 --z_size 32
@REM python generate_fractures.py --n_updates 5000 --z_size 32 --target_image_size 64


@REM ----------------------- Fracture dataset (learning rate tuning) ---------------------------------------------------------------------

@REM python vae_nca.py --n_updates 5000 --learning_rate 5e-4
@REM python generate_fractures.py --n_updates 5000 --learning_rate 5e-4
@REM python generate_fractures.py --n_updates 5000 --learning_rate 5e-4 --target_image_size 64

@REM python vae_nca.py --n_updates 5000 --learning_rate 1e-4
@REM python generate_fractures.py 


@REM -------------- Fracture dataset (beta parameter tuning with augmented dataset) ---------------------------------------------------------------------

@REM python vae_nca.py --n_updates 5000 --beta 1.0
@REM python generate_fractures.py 

@REM python vae_nca.py --n_updates 5000 --beta 1.5
@REM python generate_fractures.py --n_updates 5000 --beta 1.5
@REM python generate_fractures.py --n_updates 5000 --beta 1.5 --target_image_size 64

@REM python vae_nca.py --n_updates 5000 --beta 2.0
@REM python generate_fractures.py --n_updates 5000 --beta 2.0
@REM python generate_fractures.py --n_updates 5000 --beta 2.0 --target_image_size 64


@REM -------------- Fracture dataset (longer training with augmented dataset) ---------------------------------------------------------------------

@REM python vae_nca.py --n_updates 20000 --learning_rate 1e-4 --augment True
python generate_fractures.py --n_updates 20000 --learning_rate 1e-4 --augment True
@REM python generate_fractures.py --n_updates 20000 --learning_rate 1e-4 --augment True --target_image_size 64

@REM pause