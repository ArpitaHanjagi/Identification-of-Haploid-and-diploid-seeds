@echo off
rem Runs the remaining phases one after another. Log: results\queue_log.txt
rem Each fine-tune run resumes from its checkpoint if interrupted, so rerunning this file is safe.
cd /d E:\Identification_Haploid_Diploid
set PYTHONWARNINGS=ignore
set PY=.venv\Scripts\python.exe

:wait_classical
if exist results\finetune\classical_seed0\test_result.json goto run
timeout /t 30 /nobreak >nul
goto wait_classical

:run
echo [%date% %time%] classical seed 0 finished, starting QCNN seed 1>> results\queue_log.txt
%PY% -u 04_finetune.py --head qcnn --seed 1 --epochs 12 >> results\queue_log.txt 2>&1
echo [%date% %time%] starting QCNN seed 2>> results\queue_log.txt
%PY% -u 04_finetune.py --head qcnn --seed 2 --epochs 12 >> results\queue_log.txt 2>&1
echo [%date% %time%] evaluating (plain + TTA)>> results\queue_log.txt
%PY% -u 05_evaluate.py >> results\queue_log.txt 2>&1
echo [%date% %time%] final report>> results\queue_log.txt
%PY% -u 06_report.py >> results\queue_log.txt 2>&1
echo [%date% %time%] QUEUE DONE>> results\queue_log.txt
