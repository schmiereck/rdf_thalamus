Modify 'src/train_thalamus.py' to add a robust 'try/except' crash diagnostics block around 'loss.backward()'.

Specifically:
Replace line 159-160 (or around there):
```python
            loss = loss_dict["loss"]
            loss.backward()
```
with:
```python
            loss = loss_dict["loss"]
            try:
                loss.backward()
            except RuntimeError as e:
                print("\n=== CRASH DIAGNOSTICS ===")
                print(f"Step: {step}")
                print(f"Model: {model_type}")
                print(f"Token locus: {model.token_locus}")
                print(f"L2 locked: {model.l2_locked}")
                print(f"Loss value: {loss.item() if hasattr(loss, 'item') else loss}")
                print(f"Loss requires_grad: {loss.requires_grad if hasattr(loss, 'requires_grad') else 'No requires_grad attribute'}")
                print(f"Loss grad_fn: {loss.grad_fn if hasattr(loss, 'grad_fn') else 'No grad_fn attribute'}")
                print("\nParameter States:")
                for name, p in model.named_parameters():
                    print(f"  {name}: requires_grad={p.requires_grad}")
                print("==========================\n")
                raise e
```

Then run '.venv/Scripts/python.exe src/train_thalamus.py --model gated --seed 456' to reproduce the crash and show the diagnostics printout.