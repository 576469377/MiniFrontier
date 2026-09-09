# Research preview experiment snapshot

Captured: 2026-09-09T08:20:06.312154+00:00

Numeric records only. Running trials are partial; diagnostics measure learnability, not chat quality.

| Experiment | State | CE tokens | Latest validation LM NLL | Measured CE/s |
|---|---|---:|---:|---:|
| [strategy-recipe-pilots-v2--minideepseekv4--adamw-20m](strategy-recipe-pilots-v2--minideepseekv4--adamw-20m.json) ([CSV](strategy-recipe-pilots-v2--minideepseekv4--adamw-20m.csv)) | running | 13901894 | 5.855686939924099 | 526.3428746014318 |
| [strategy-recipe-pilots-v2--minideepseekv4--muon-20m](strategy-recipe-pilots-v2--minideepseekv4--muon-20m.json) ([CSV](strategy-recipe-pilots-v2--minideepseekv4--muon-20m.csv)) | complete | 20013084 | 5.334631086140065 | 516.8445481960501 |
| [strategy-recipe-pilots-v2--minikimik3--adamw-20m](strategy-recipe-pilots-v2--minikimik3--adamw-20m.json) ([CSV](strategy-recipe-pilots-v2--minikimik3--adamw-20m.csv)) | running | 17021901 | 5.632155408497192 | 570.1488719772458 |
| [strategy-recipe-pilots-v2--minikimik3--muon-20m](strategy-recipe-pilots-v2--minikimik3--muon-20m.json) ([CSV](strategy-recipe-pilots-v2--minikimik3--muon-20m.csv)) | complete | 20010912 | 5.25667633755248 | 561.3834668387981 |
| [strategy-recipe-pilots-v2--miniqwen4-after-q0-extension--muon-20m](strategy-recipe-pilots-v2--miniqwen4-after-q0-extension--muon-20m.json) ([CSV](strategy-recipe-pilots-v2--miniqwen4-after-q0-extension--muon-20m.csv)) | running | 17873675 | 5.230578570976059 | 287.222375773371 |
| [strategy-diagnostics-v2--minideepseekv4](strategy-diagnostics-v2--minideepseekv4.json) ([CSV](strategy-diagnostics-v2--minideepseekv4.csv)) | complete | 500735 | 1.452930539449056 | 163.5091131107459 |
| [strategy-diagnostics-v2--minikimik3](strategy-diagnostics-v2--minikimik3.json) ([CSV](strategy-diagnostics-v2--minikimik3.csv)) | complete | 500450 | 1.1787232716878255 | 135.16564801724988 |
| [strategy-diagnostics-v2--miniqwen4](strategy-diagnostics-v2--miniqwen4.json) ([CSV](strategy-diagnostics-v2--miniqwen4.csv)) | complete | 500450 | 1.1579292233784992 | 55.71579888290027 |
| [strategy-diagnostics-v2--miniqwen4-extension-1m](strategy-diagnostics-v2--miniqwen4-extension-1m.json) ([CSV](strategy-diagnostics-v2--miniqwen4-extension-1m.csv)) | complete | 500450 | 1.254478308359782 | 112.72600286013086 |

Commands retain `${WORKSPACE}` as an explicit binding. Use the recorded source commit and data/tokenizer hashes.
A matching dataset must be reconstructed from its pinned preparation recipe; this snapshot does not redistribute the dataset.
See the project experiment guide for failure reports, limitations and reconstruction instructions.
