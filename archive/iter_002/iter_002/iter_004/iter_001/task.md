Modify the file `src/environment.py` to support seeding in `PhysicsSandbox`.
Specifically:
1. In `PhysicsSandbox.__init__`, add `seed=None` parameter: `def __init__(self, N=2, substeps=10, sigma_blur=0.5, seed=None):`
2. Store `self.seed = seed` in `__init__`.
3. In `__init__`, pass the seed to `self.reset(seed=seed)` instead of calling `self.reset()`.
4. In `PhysicsSandbox.reset`, add `seed=None` parameter: `def reset(self, seed=None):`
5. At the beginning of `reset`, if `seed is not None`, call `np.random.seed(seed)` and set `self.seed = seed`.
6. Run `python src/test_integration.py` to make sure the tests pass completely and there are no regressions.
7. Return a summary of the edits and the result of the verification test.