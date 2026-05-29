# Iter_027 Separate z_dyn Encoder Architectural Probe — Analysis
**Dual collapse criterion:** collapsed = collapsed_eval OR collapsed_train
**Sanity disqualification:** final_train_loss > 50 counted as collapsed
**Arms:** Aprime (shared, centroid_gated), A (shared, mean), B (separate, JEPA+VICReg), C (separate, VICReg-only)
**Gate threshold:** ≤10% collapse rate (dual criterion)

---
## Per-Arm Summary
### Aprime (shared, centroid_gated)
- N seeds: 10
- Collapse rate (dual): 0.30 (3/10)
- Collapse rate (eval-only): 0.30 (3/10)
- Collapse rate (train-only): 0.20 (2/10)
- Mean final train loss: 4.7470 +/- 5.6087
- Centroid MSE (REF ONLY): 174.86
- delta_R2_color (REF ONLY): -0.0158
- Mean abs corr: 0.318
- Parameter count: 81368

### A (shared, mean readout)
- N seeds: 10
- Collapse rate (dual): 0.40 (4/10)
- Collapse rate (eval-only): 0.40 (4/10)
- Collapse rate (train-only): 0.20 (2/10)
- Mean final train loss: 8.7341 +/- 9.2575
- Centroid MSE (REF ONLY): 170.22
- delta_R2_color (REF ONLY): 0.1469
- Mean abs corr: 0.418
- Parameter count: 80336

### B (separate, JEPA+VICReg)
- N seeds: 10
- Collapse rate (dual): 0.30 (3/10)
- Collapse rate (eval-only): 0.30 (3/10)
- Collapse rate (train-only): 0.30 (3/10)
- Mean final train loss: 6.9755 +/- 8.3202
- Centroid MSE (REF ONLY): 136.86
- delta_R2_color (REF ONLY): -0.0171
- Mean abs corr: 0.407
- Parameter count: 135608

### C (separate, VICReg-only on z_dyn)
- N seeds: 10
- Collapse rate (dual): 0.00 (0/10)
- Collapse rate (eval-only): 0.00 (0/10)
- Collapse rate (train-only): 0.00 (0/10)
- Mean final train loss: 3.1901 +/- 1.3971
- Centroid MSE (REF ONLY): 155.96
- delta_R2_color (REF ONLY): 0.1812
- Mean abs corr: 0.210
- Parameter count: 135608

## Per-Seed Train-vs-Eval Std Gap Table (CO-EQUAL reporting)

| seed | arm | collapsed_eval | collapsed_train | collapsed | per_dim_std_eval | per_dim_std_train |
|------|-----|----------------|-----------------|-----------|------------------|-------------------|
| 31 | Aprime (shared, centroid_gated) | N | N | N | [0.6900812983512878, 0.5087348222732544, 0.8297139406204224] | [1.044479489326477, 1.132667899131775, 1.0485161542892456] |
| 53 | Aprime (shared, centroid_gated) | Y | Y | Y | [1.091073989868164, 0.21652770042419434, 1.0237195491790771] | [1.0817017555236816, 0.21109184622764587, 1.1770278215408325] |
| 71 | Aprime (shared, centroid_gated) | Y | Y | Y | [0.1435619294643402, 0.13642893731594086, 0.7756352424621582] | [0.26036587357521057, 0.24234260618686676, 1.2084033489227295] |
| 17 | Aprime (shared, centroid_gated) | N | N | N | [0.8907298445701599, 0.9074664115905762, 0.8128207921981812] | [0.9799031019210815, 1.2155483961105347, 1.1516410112380981] |
| 83 | Aprime (shared, centroid_gated) | N | N | N | [0.610019862651825, 0.8798079490661621, 1.013811469078064] | [1.135209083557129, 1.2979122400283813, 0.993702232837677] |
| 113 | Aprime (shared, centroid_gated) | N | N | N | [0.9309724569320679, 0.7023010849952698, 0.5631149411201477] | [1.0469417572021484, 1.1297504901885986, 1.0691190958023071] |
| 7 | Aprime (shared, centroid_gated) | N | N | N | [1.1022812128067017, 1.1928117275238037, 0.9431415796279907] | [1.247782826423645, 1.0701547861099243, 1.0813584327697754] |
| 97 | Aprime (shared, centroid_gated) | N | N | N | [1.3883651494979858, 1.3919919729232788, 0.5290884375572205] | [1.1423726081848145, 1.1891125440597534, 1.0238968133926392] |
| 31 | A (shared, mean readout) | N | N | N | [0.7403698563575745, 1.0383269786834717, 1.0200731754302979] | [1.0895251035690308, 1.15957510471344, 1.0279890298843384] |
| 53 | A (shared, mean readout) | Y | Y | Y | [0.4815291166305542, 0.4329003095626831, 0.29009437561035156] | [0.6117871999740601, 0.5388663411140442, 0.383466899394989] |
| 71 | A (shared, mean readout) | Y | Y | Y | [0.522251307964325, 0.5416331887245178, 0.42197105288505554] | [0.5515841841697693, 0.5716106295585632, 0.49074897170066833] |
| 7 | A (shared, mean readout) | Y | N | Y | [0.42017143964767456, 0.5868954658508301, 0.7522459030151367] | [0.529568076133728, 0.7133768796920776, 1.0328096151351929] |
| 17 | A (shared, mean readout) | N | N | N | [1.290990948677063, 0.5661392211914062, 0.7887841463088989] | [1.1396924257278442, 0.8834198713302612, 0.996010422706604] |
| 83 | A (shared, mean readout) | Y | N | Y | [0.5599529147148132, 0.45002973079681396, 1.4919637441635132] | [1.1014496088027954, 1.0610147714614868, 1.0429844856262207] |
| 127 | Aprime (shared, centroid_gated) | N | N | N | [0.8028019666671753, 0.8675133585929871, 0.944101870059967] | [1.1500694751739502, 1.0964062213897705, 1.1238296031951904] |
| 149 | Aprime (shared, centroid_gated) | Y | N | Y | [0.4813540279865265, 0.3430708050727844, 0.5109449028968811] | [1.1962615251541138, 1.1690011024475098, 1.1069741249084473] |
| 97 | A (shared, mean readout) | N | N | N | [0.8162828683853149, 0.6641883254051208, 1.1972523927688599] | [0.9927791357040405, 1.1341181993484497, 1.1924128532409668] |
| 113 | A (shared, mean readout) | N | N | N | [0.6005437970161438, 1.4233700037002563, 1.1023433208465576] | [1.051513433456421, 1.126657247543335, 1.0059555768966675] |
| 127 | A (shared, mean readout) | N | N | N | [1.7285139560699463, 0.9708102345466614, 0.6668726205825806] | [1.0472412109375, 0.9656734466552734, 1.1839048862457275] |
| 149 | A (shared, mean readout) | N | N | N | [0.7632652521133423, 1.0257560014724731, 0.6207776069641113] | [1.0283453464508057, 0.8968602418899536, 1.2177058458328247] |
| 7 | B (separate, JEPA+VICReg) | N | N | N | [0.936988115310669, 0.6826592087745667, 0.8245308995246887] | [1.1371022462844849, 1.153149127960205, 1.102337121963501] |
| 17 | B (separate, JEPA+VICReg) | Y | Y | Y | [0.25608718395233154, 0.5880433917045593, 0.7221770286560059] | [0.3258027136325836, 0.9396639466285706, 0.9417451024055481] |
| 31 | B (separate, JEPA+VICReg) | N | N | N | [0.7568902373313904, 1.1060477495193481, 0.7516408562660217] | [1.0755078792572021, 1.101315975189209, 1.184220790863037] |
| 53 | B (separate, JEPA+VICReg) | Y | Y | Y | [0.14099344611167908, 0.15847276151180267, 0.1467740386724472] | [0.45330747961997986, 0.5017197728157043, 0.45763689279556274] |
| 113 | B (separate, JEPA+VICReg) | N | N | N | [0.8522432446479797, 1.4158908128738403, 1.04795503616333] | [1.069693922996521, 1.2574657201766968, 1.0920424461364746] |
| 83 | B (separate, JEPA+VICReg) | N | N | N | [0.9963605403900146, 1.0209060907363892, 1.1401512622833252] | [1.3144810199737549, 1.1549501419067383, 1.0896575450897217] |
| 71 | B (separate, JEPA+VICReg) | Y | Y | Y | [1.539145588874817, 0.11855988949537277, 0.5593976378440857] | [1.2764018774032593, 0.22084450721740723, 0.9871507287025452] |
| 97 | B (separate, JEPA+VICReg) | N | N | N | [0.6577274203300476, 1.1256494522094727, 0.7661643624305725] | [1.1336747407913208, 1.0718848705291748, 0.9929403066635132] |
| 127 | B (separate, JEPA+VICReg) | N | N | N | [0.7632887363433838, 1.6750543117523193, 0.6088494658470154] | [0.9720712900161743, 1.1007248163223267, 1.1512418985366821] |
| 149 | B (separate, JEPA+VICReg) | N | N | N | [0.5413229465484619, 0.9923276305198669, 0.8427644371986389] | [1.1441642045974731, 1.1294265985488892, 1.1439547538757324] |
| 7 | C (separate, VICReg-only on z_dyn) | N | N | N | [1.025480031967163, 0.6992092132568359, 1.2230420112609863] | [1.1739524602890015, 1.2019647359848022, 0.9832931756973267] |
| 17 | C (separate, VICReg-only on z_dyn) | N | N | N | [0.7958140969276428, 0.9825933575630188, 1.1159307956695557] | [1.06849205493927, 1.0379663705825806, 1.113615870475769] |
| 31 | C (separate, VICReg-only on z_dyn) | N | N | N | [0.7215263247489929, 1.1757714748382568, 1.0733572244644165] | [1.0684863328933716, 1.0327187776565552, 1.0205610990524292] |
| 53 | C (separate, VICReg-only on z_dyn) | N | N | N | [1.0113133192062378, 1.1661514043807983, 0.5023847818374634] | [1.19966721534729, 1.089213490486145, 1.0021897554397583] |
| 71 | C (separate, VICReg-only on z_dyn) | N | N | N | [1.072725772857666, 1.3348945379257202, 0.9762439727783203] | [1.0900534391403198, 1.1256523132324219, 1.1201348304748535] |
| 83 | C (separate, VICReg-only on z_dyn) | N | N | N | [1.034875512123108, 1.0132226943969727, 0.9091587662696838] | [1.0909723043441772, 0.9933547973632812, 1.1186137199401855] |
| 97 | C (separate, VICReg-only on z_dyn) | N | N | N | [0.90486079454422, 1.0403128862380981, 1.109036922454834] | [1.1426352262496948, 0.9773420095443726, 1.2142828702926636] |
| 113 | C (separate, VICReg-only on z_dyn) | N | N | N | [1.2928048372268677, 1.1435081958770752, 0.8959451913833618] | [1.2045356035232544, 1.0242048501968384, 1.1283220052719116] |
| 127 | C (separate, VICReg-only on z_dyn) | N | N | N | [1.2918542623519897, 1.2501122951507568, 1.4641244411468506] | [1.1544636487960815, 1.0337274074554443, 1.078598976135254] |
| 149 | C (separate, VICReg-only on z_dyn) | N | N | N | [1.1526236534118652, 1.0285224914550781, 1.3112367391586304] | [1.1168444156646729, 1.0400534868240356, 1.2031984329223633] |

## Gate Check

- **Aprime (shared, centroid_gated):** FAIL (collapse rate 0.30)
- **A (shared, mean readout):** FAIL (collapse rate 0.40)
- **B (separate, JEPA+VICReg):** FAIL (collapse rate 0.30)
- **C (separate, VICReg-only on z_dyn):** PASS (collapse rate 0.00)

## Pre-Registered Outcome Classification

Arm B collapse rate: 0.30
Arm C collapse rate: 0.00

Outcome: SECOND NULL. Arm B ≥20%. Shared backbone is not the primary structural cause.
Project pivots per Manager instruction.

## Readout Effect (Aprime vs A)

- Aprime (centroid_gated, shared): collapse rate 0.30
- A (mean, shared): collapse rate 0.40
- Difference: +0.10 — readout type affects collapse

## Parameter Count Comparison

- Aprime (shared, centroid_gated): 81368
- A (shared, mean readout): 80336
- B (separate, JEPA+VICReg): 135608
- C (separate, VICReg-only on z_dyn): 135608

Capacity confound note: If Arm B passes, the result is consistent with gradient decoupling but also consistent with added capacity (Arm B roughly doubles encoder parameters). A capacity-matched shared-backbone control is the mandatory iter_028 follow-up.
