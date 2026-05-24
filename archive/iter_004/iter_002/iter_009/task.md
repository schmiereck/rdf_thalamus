Execute the 5-seed systematic evaluation campaign by running each of the 15 experiments using a separate bash command.

Please invoke '.venv/Scripts/python.exe src/train_thalamus.py --model <model> --seed <seed>' for all 15 configurations sequentially. 

List of 15 commands to run:
1. .venv/Scripts/python.exe src/train_thalamus.py --model gated --seed 42
2. .venv/Scripts/python.exe src/train_thalamus.py --model gated --seed 123
3. .venv/Scripts/python.exe src/train_thalamus.py --model gated --seed 456
4. .venv/Scripts/python.exe src/train_thalamus.py --model gated --seed 789
5. .venv/Scripts/python.exe src/train_thalamus.py --model gated --seed 999

6. .venv/Scripts/python.exe src/train_thalamus.py --model b1 --seed 42
7. .venv/Scripts/python.exe src/train_thalamus.py --model b1 --seed 123
8. .venv/Scripts/python.exe src/train_thalamus.py --model b1 --seed 456
9. .venv/Scripts/python.exe src/train_thalamus.py --model b1 --seed 789
10. .venv/Scripts/python.exe src/train_thalamus.py --model b1 --seed 999

11. .venv/Scripts/python.exe src/train_thalamus.py --model nongated --seed 42
12. .venv/Scripts/python.exe src/train_thalamus.py --model nongated --seed 123
13. .venv/Scripts/python.exe src/train_thalamus.py --model nongated --seed 456
14. .venv/Scripts/python.exe src/train_thalamus.py --model nongated --seed 789
15. .venv/Scripts/python.exe src/train_thalamus.py --model nongated --seed 999

Verify that all 15 experiments complete successfully with exit code 0. Do not compile the results yet; we will do that in the next step. Just run the 15 experiments sequentially and show the output logs.