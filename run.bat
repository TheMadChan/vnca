@echo off

@REM python vae-nca.py --dataset mnist --n_updates 10 --batch_size 32 --test_batch_size 32 --z_size 64 --learning_rate 1e-4 
@REM python vae-nca.py --dataset mnist --n_updates 1000 --batch_size 32 --test_batch_size 32 --z_size 16 --learning_rate 1e-4 
@REM python vae-nca.py --dataset mnist --n_updates 1000 --batch_size 32 --test_batch_size 32 --z_size 64 --learning_rate 1e-4 
@REM python vae-nca.py --dataset mnist --n_updates 1000 --batch_size 32 --test_batch_size 32 --z_size 128 --learning_rate 1e-4 
@REM python vae-nca.py --dataset mnist --n_updates 1000 --batch_size 32 --test_batch_size 32 --z_size 256 --learning_rate 1e-4 
@REM python vae-nca.py --dataset mnist --n_updates 1000 --batch_size 64 --test_batch_size 32 --z_size 64 --learning_rate 1e-4 

@REM python vae-nca.py --n_updates 10000 --batch_size 64 --test_batch_size 32 --z_size 128 --learning_rate 1e-4 --bin_threshold 0.75 
python vae-nca.py --n_updates 10000 --batch_size 32 --test_batch_size 32 --z_size 128 --learning_rate 5e-4 --bin_threshold 0.75 
@REM python vae-nca.py --n_updates 5000 --batch_size 32 --test_batch_size 32 --z_size 128 --learning_rate 1e-4 --bin_threshold 0.5 

@REM pause