# GraphPAN_advection_sparse

Train and test model on advection data:

```
python main.py --dataset advection --sparse 1 --learning_rate 0.001 --symmetric 0 --n 2304 --num_epochs 200 --message_passing_steps 3
```

```
python main.py --dataset random --sparse 1 --learning_rate 0.001 --symmetric 0 --n 100 --num_epochs 2 --message_passing_steps 3
```
