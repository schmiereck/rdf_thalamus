# Iter_028 Shared-Backbone mask_dyn_sim Probe — Analysis
**Dual collapse criterion:** collapsed = collapsed_eval OR collapsed_train
**Sanity disqualification:** final_train_loss > 50 counted as collapsed
**Arms:** D0 (baseline), C1 (primary), C2 (seed robustness), C3 (weight robustness)
**Gate threshold:** ≤10% collapse rate (dual criterion)
**Relative threshold (F4/H2):** D0's ΔR² and mean_abs_corr are in-iteration null references

---
## Per-Arm Summary
### D0 (baseline replication)
- N seeds: 10
- Collapse rate (dual): 0.30 (3/10)
- Collapse rate (eval-only): 0.30 (3/10)
- Collapse rate (train-only): 0.10 (1/10)
- **PRIMARY collapse rate (excl. timeouts):** 0.30 (3/10)
- **Timeout count:** 0
- Mean final train loss: 7.1821 +/- 5.3140
- Centroid MSE (REF ONLY): 113.64
- delta_R2_color (REF ONLY): 0.0541
- Mean abs corr: 0.999
- Parameter count: 80336

### C1 (primary, mask_dyn_sim)
- N seeds: 10
- Collapse rate (dual): 0.20 (2/10)
- Collapse rate (eval-only): 0.20 (2/10)
- Collapse rate (train-only): 0.20 (2/10)
- **PRIMARY collapse rate (excl. timeouts):** 0.20 (2/10)
- **Timeout count:** 0
- Mean final train loss: 5.9320 +/- 6.0804
- Centroid MSE (REF ONLY): 116.55
- delta_R2_color (REF ONLY): 0.2308
- Mean abs corr: 0.521
- Parameter count: 80336

### C2 (seed robustness)
- N seeds: 10
- Collapse rate (dual): 0.00 (0/10)
- Collapse rate (eval-only): 0.00 (0/10)
- Collapse rate (train-only): 0.00 (0/10)
- **PRIMARY collapse rate (excl. timeouts):** 0.00 (0/10)
- **Timeout count:** 0
- Mean final train loss: 3.0419 +/- 0.7447
- Centroid MSE (REF ONLY): 99.68
- delta_R2_color (REF ONLY): 0.5135
- Mean abs corr: 0.435
- Parameter count: 80336

### C3 (weight robustness)
- N seeds: 10
- Collapse rate (dual): 0.20 (2/10)
- Collapse rate (eval-only): 0.20 (2/10)
- Collapse rate (train-only): 0.20 (2/10)
- **PRIMARY collapse rate (excl. timeouts):** 0.20 (2/10)
- **Timeout count:** 0
- Mean final train loss: 6.1574 +/- 6.2191
- Centroid MSE (REF ONLY): 115.95
- delta_R2_color (REF ONLY): 0.1677
- Mean abs corr: 0.474
- Parameter count: 80336

## Per-Seed Train-vs-Eval Std Gap Table (CO-EQUAL reporting)

| seed | arm | collapsed_eval | collapsed_train | collapsed | per_dim_std_eval | per_dim_std_train |
|------|-----|----------------|-----------------|-----------|------------------|-------------------|
| 17 | C1 (primary, mask_dyn_sim) | N | N | N | [1.1207748651504517, 1.013479232788086, 0.9700503349304199] | [1.2518445253372192, 1.1362051963806152, 1.2581039667129517] |
| 31 | C1 (primary, mask_dyn_sim) | N | N | N | [1.4032409191131592, 0.9551020860671997, 1.4063873291015625] | [1.2264087200164795, 1.2042049169540405, 1.1349748373031616] |
| 7 | C1 (primary, mask_dyn_sim) | N | N | N | [1.1822550296783447, 0.7491927742958069, 0.5265175104141235] | [1.1742380857467651, 1.1118440628051758, 1.0650782585144043] |
| 113 | D0 (baseline replication) | N | N | N | [0.9430386424064636, 0.9462921619415283, 0.9483147263526917] | [1.0231280326843262, 1.0208537578582764, 1.0205248594284058] |
| 127 | D0 (baseline replication) | N | N | N | [0.5586563348770142, 0.5430616140365601, 0.5459471344947815] | [1.054928183555603, 1.0523148775100708, 1.0522698163986206] |
| 149 | D0 (baseline replication) | N | N | N | [0.6771717667579651, 0.68696528673172, 0.6861668825149536] | [1.0928425788879395, 1.0940412282943726, 1.0935845375061035] |
| 17 | D0 (baseline replication) | Y | N | Y | [0.45142626762390137, 0.4545062780380249, 0.45183753967285156] | [0.8975547552108765, 0.898200511932373, 0.8967897295951843] |
| 31 | D0 (baseline replication) | N | N | N | [0.9379981756210327, 0.948223888874054, 0.9389623403549194] | [0.9865886569023132, 0.9854623079299927, 0.9809343218803406] |
| 53 | D0 (baseline replication) | Y | Y | Y | [0.2858414053916931, 0.31571927666664124, 0.2807048559188843] | [0.2791413962841034, 0.30108529329299927, 0.27185943722724915] |
| 7 | D0 (baseline replication) | N | N | N | [0.5050872564315796, 0.5058268308639526, 0.5031935572624207] | [1.0632712841033936, 1.0651708841323853, 1.0665079355239868] |
| 71 | D0 (baseline replication) | N | N | N | [1.0255322456359863, 1.0369428396224976, 1.0426826477050781] | [1.0946100950241089, 1.097541332244873, 1.0948560237884521] |
| 83 | D0 (baseline replication) | Y | N | Y | [0.40898212790489197, 0.41012391448020935, 0.4092220067977905] | [0.9519149661064148, 0.9508026242256165, 0.9566674828529358] |
| 97 | D0 (baseline replication) | N | N | N | [1.0079092979431152, 1.048653244972229, 1.0456942319869995] | [1.0966451168060303, 1.0678714513778687, 1.0639780759811401] |
| 71 | C1 (primary, mask_dyn_sim) | Y | Y | Y | [0.011793926358222961, 0.011943905614316463, 0.011722876690328121] | [0.013180590234696865, 0.01338714174926281, 0.013135334476828575] |
| 83 | C1 (primary, mask_dyn_sim) | N | N | N | [0.6277302503585815, 0.6791201233863831, 0.7042272090911865] | [0.9921887516975403, 0.9897155165672302, 0.9878445267677307] |
| 53 | C1 (primary, mask_dyn_sim) | Y | Y | Y | [0.009528139606118202, 0.00892987847328186, 0.008524000644683838] | [0.009408957324922085, 0.009378931485116482, 0.009548348374664783] |
| 97 | C1 (primary, mask_dyn_sim) | N | N | N | [1.1769721508026123, 1.1341408491134644, 0.798080563545227] | [1.3132518529891968, 1.048679232597351, 1.256658673286438] |
| 113 | C1 (primary, mask_dyn_sim) | N | N | N | [1.2221943140029907, 1.4379502534866333, 1.5583457946777344] | [1.295485496520996, 1.1055678129196167, 1.2976877689361572] |
| 127 | C1 (primary, mask_dyn_sim) | N | N | N | [1.2889305353164673, 1.3365734815597534, 1.0954170227050781] | [0.9696534276008606, 0.9739397764205933, 1.0750168561935425] |
| 149 | C1 (primary, mask_dyn_sim) | N | N | N | [0.6842865347862244, 0.8091272115707397, 1.144155502319336] | [1.2973533868789673, 1.2003144025802612, 1.1746714115142822] |
| 101 | C2 (seed robustness) | N | N | N | [1.212720274925232, 1.2104169130325317, 1.3044859170913696] | [1.2375487089157104, 1.1939889192581177, 1.161797285079956] |
| 103 | C2 (seed robustness) | N | N | N | [1.335146188735962, 1.492596983909607, 1.1682548522949219] | [1.1814237833023071, 1.1515703201293945, 1.1541004180908203] |
| 109 | C2 (seed robustness) | N | N | N | [0.7485653162002563, 1.3030494451522827, 1.030604600906372] | [1.1440976858139038, 1.2323588132858276, 1.4566434621810913] |
| 131 | C2 (seed robustness) | N | N | N | [1.1942037343978882, 1.235099196434021, 1.4522252082824707] | [1.1859413385391235, 1.3865395784378052, 1.1775387525558472] |
| 107 | C2 (seed robustness) | N | N | N | [0.9340779185295105, 0.925714910030365, 0.9002575874328613] | [1.0571396350860596, 1.0530803203582764, 1.1641428470611572] |
| 137 | C2 (seed robustness) | N | N | N | [0.8740630745887756, 1.2618876695632935, 1.062487006187439] | [1.1649261713027954, 1.212328314781189, 1.259137749671936] |
| 139 | C2 (seed robustness) | N | N | N | [0.8611564040184021, 0.7306297421455383, 0.9287903904914856] | [1.0774028301239014, 0.9904107451438904, 0.9949094653129578] |
| 151 | C2 (seed robustness) | N | N | N | [0.7387563586235046, 0.7675598859786987, 0.7788891196250916] | [1.1228917837142944, 1.091412901878357, 1.0909028053283691] |
| 157 | C2 (seed robustness) | N | N | N | [1.0822136402130127, 1.034249186515808, 0.53672194480896] | [1.1395050287246704, 1.0957475900650024, 1.1150872707366943] |
| 163 | C2 (seed robustness) | N | N | N | [1.1301568746566772, 1.3089649677276611, 1.4288692474365234] | [1.093747854232788, 1.3066173791885376, 1.0738894939422607] |
| 7 | C3 (weight robustness) | N | N | N | [1.1572352647781372, 0.9154746532440186, 0.5511087775230408] | [1.2052713632583618, 1.1963671445846558, 1.1454334259033203] |
| 17 | C3 (weight robustness) | N | N | N | [0.9946552515029907, 0.9817374348640442, 1.1344002485275269] | [1.1669590473175049, 1.1059176921844482, 1.1883317232131958] |
| 31 | C3 (weight robustness) | N | N | N | [1.3959176540374756, 0.954413115978241, 1.4090490341186523] | [1.2021719217300415, 1.1620458364486694, 1.1140258312225342] |
| 53 | C3 (weight robustness) | Y | Y | Y | [0.010474135167896748, 0.01034002099186182, 0.01033396553248167] | [0.016611235216259956, 0.01632840558886528, 0.016534244641661644] |
| 71 | C3 (weight robustness) | Y | Y | Y | [0.014668826013803482, 0.014267642050981522, 0.014743699692189693] | [0.015819452702999115, 0.015536324121057987, 0.015640921890735626] |
| 83 | C3 (weight robustness) | N | N | N | [0.6338580846786499, 0.6707978844642639, 0.644975483417511] | [0.9755149483680725, 0.9845852851867676, 0.9786364436149597] |
| 97 | C3 (weight robustness) | N | N | N | [1.1792516708374023, 1.1128199100494385, 0.9149940013885498] | [1.3774540424346924, 1.0827299356460571, 1.2500087022781372] |
| 113 | C3 (weight robustness) | N | N | N | [1.2475361824035645, 1.4562163352966309, 1.5101746320724487] | [1.3462494611740112, 1.1236286163330078, 1.2848347425460815] |
| 127 | C3 (weight robustness) | N | N | N | [1.6043181419372559, 1.0740092992782593, 1.0148371458053589] | [1.143549919128418, 1.0548510551452637, 1.3229542970657349] |
| 149 | C3 (weight robustness) | N | N | N | [0.7387350797653198, 0.8220909237861633, 1.1172311305999756] | [1.3251488208770752, 1.1540974378585815, 1.146579623222351] |

## Gate Check (PRIMARY — excluding timeouts)

- **D0 (baseline replication):** FAIL (PRIMARY collapse rate 0.30, n=10)
- **C1 (primary, mask_dyn_sim):** FAIL (PRIMARY collapse rate 0.20, n=10)
- **C2 (seed robustness):** PASS (PRIMARY collapse rate 0.00, n=10)
- **C3 (weight robustness):** FAIL (PRIMARY collapse rate 0.20, n=10)

## D0 vs C1 — Independent Readout Comparison (Relative-Threshold Gate)

**D0 (null reference):**
- Mean ΔR²_color: 0.0541
- Mean mean_abs_corr: 0.9995

**C1 (primary):**
- Mean ΔR²_color: 0.2308
- Mean mean_abs_corr: 0.5211

**Relative thresholds:**
- ΔR² gate: C1 ≥ D0 + 0.05 = 0.1041; C1 = 0.2308 → PASS
- mean_abs_corr gate: C1 ≤ D0 + 0.05 = 1.0495; C1 = 0.5211 → PASS

H2 relative gate (ΔR² AND mean_abs_corr): PASS
## Pre-Registered Outcome Classification

C1 **PRIMARY** collapse rate (excl. timeouts): 0.20
C2 **PRIMARY** collapse rate (excl. timeouts): 0.00
C3 **PRIMARY** collapse rate (excl. timeouts): 0.20

**F1:** FALSIFIED — mask_dyn_sim alone insufficient on the shared backbone.
**F3:** NOT ROBUST — C1 result sensitive to weight perturbation.

## Sample-Size Caveat

Fisher's exact test for 0/10 vs 3/10 gives p ≈ 0.21; the design cannot formally distinguish 0% from 10–20% at this sample size. Results are reported as point estimates with this limit explicitly noted.

## Parameter Count Comparison

- D0 (baseline replication): 80336
- C1 (primary, mask_dyn_sim): 80336
- C2 (seed robustness): 80336
- C3 (weight robustness): 80336

## Timeout Audit

Timeouts are ENGINEERING failures, NOT representation failures.
The PRIMARY collapse rate (used for gates) excludes timeouts.

- **D0 (baseline replication):** 0 timeout(s) → OK
- **C1 (primary, mask_dyn_sim):** 0 timeout(s) → OK
- **C2 (seed robustness):** 0 timeout(s) → OK
- **C3 (weight robustness):** 0 timeout(s) → OK

**Overall interpretable:** Yes

## Language Constraints (from pre-registration)

Results are reported using 'does not destabilize VICReg-maintained variance' or 'is consistent with.' Terms such as 'breakthrough', 'causal driver', 'eliminated', 'BEST', 'proves', 'demonstrates', or 'resolves' are NOT used.
Even a fully-confirmed C1+C2+C3 result is reported per the constructional framing in the hypothesis, not as 'sim_loss_dyn causes collapse.'
