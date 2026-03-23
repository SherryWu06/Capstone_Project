# SLURM job logs

Each `sbatch run.sh` writes **unique** log files (nothing overwrites):

- `slurm/job_<JOBID>.out` — stdout (including `time` output)
- `slurm/job_<JOBID>.err` — stderr

Find `<JOBID>` in the `sbatch` confirmation line, or:

```bash
ls -lt slurm/
```

## Optional: include a wall-clock stamp in the filename

`#SBATCH` lines cannot use `$(date)`. From the login node, override paths when submitting:

```bash
TS=$(date +%Y%m%d_%H%M%S)
sbatch -o "slurm/job_${TS}_%j.out" -e "slurm/job_${TS}_%j.err" run.sh
```

Slurm still replaces `%j` with the job ID after submission.
